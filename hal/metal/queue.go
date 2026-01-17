// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"unsafe"

	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

// Queue implements hal.Queue for Metal.
type Queue struct {
	device       *Device
	commandQueue ID // id<MTLCommandQueue>
}

// Submit submits command buffers to the GPU.
func (q *Queue) Submit(commandBuffers []hal.CommandBuffer, fence hal.Fence, fenceValue uint64) error {
	for _, buf := range commandBuffers {
		cb, ok := buf.(*CommandBuffer)
		if !ok || cb == nil {
			continue
		}

		// If fence provided, signal it on completion
		if fence != nil {
			if mtlFence, ok := fence.(*Fence); ok && mtlFence != nil {
				// Metal uses MTLEvent for synchronization
				// encodeSignalEvent:value: on command buffer
				_ = MsgSend(cb.raw, Sel("encodeSignalEvent:value:"),
					uintptr(mtlFence.event), uintptr(fenceValue))
				mtlFence.value = fenceValue
			}
		}

		// Schedule presentation BEFORE commit (Metal requirement)
		if cb.drawable != 0 {
			_ = MsgSend(cb.raw, Sel("presentDrawable:"), uintptr(cb.drawable))
		}

		// Commit the command buffer
		_ = MsgSend(cb.raw, Sel("commit"))
	}
	return nil
}

// WriteBuffer writes data to a buffer immediately.
func (q *Queue) WriteBuffer(buffer hal.Buffer, offset uint64, data []byte) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil {
		return
	}
	if len(data) == 0 {
		return
	}
	if offset >= buf.size {
		return
	}
	if uint64(len(data)) > buf.size-offset {
		return
	}

	ptr := buf.Contents()
	if ptr != 0 {
		// Copy data using unsafe for mapped buffers.
		dst := unsafe.Slice((*byte)(unsafe.Pointer(ptr+uintptr(offset))), len(data))
		copy(dst, data)
		return
	}

	if buf.usage&types.BufferUsageCopyDst == 0 {
		return
	}

	staging, err := q.device.CreateBuffer(&hal.BufferDescriptor{
		Label: "buffer-write-staging",
		Size:  uint64(len(data)),
		Usage: types.BufferUsageCopySrc | types.BufferUsageMapWrite,
	})
	if err != nil {
		return
	}
	defer q.device.DestroyBuffer(staging)

	stagingBuf, ok := staging.(*Buffer)
	if !ok || stagingBuf == nil {
		return
	}
	stagingPtr := stagingBuf.Contents()
	if stagingPtr == 0 {
		return
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(stagingPtr)), len(data)), data)

	encoder, err := q.device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{
		Label: "buffer-write-encoder",
	})
	if err != nil {
		return
	}
	encoder.CopyBufferToBuffer(staging, buffer, []hal.BufferCopy{
		{
			SrcOffset: 0,
			DstOffset: offset,
			Size:      uint64(len(data)),
		},
	})
	cmdBuffer, err := encoder.EndEncoding()
	if err != nil {
		return
	}
	_ = q.Submit([]hal.CommandBuffer{cmdBuffer}, nil, 0)
	if cb, ok := cmdBuffer.(*CommandBuffer); ok && cb != nil {
		_ = MsgSend(cb.raw, Sel("waitUntilCompleted"))
		cb.Destroy()
	}
}

// WriteTexture writes data to a texture immediately.
func (q *Queue) WriteTexture(dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) {
	if dst == nil || dst.Texture == nil || layout == nil || size == nil || len(data) == 0 {
		return
	}

	tex, ok := dst.Texture.(*Texture)
	if !ok || tex == nil {
		return
	}

	if size.Width == 0 || size.Height == 0 {
		return
	}

	bytesPerRow := layout.BytesPerRow
	if bytesPerRow == 0 {
		bytesPerTexel := bytesPerTexel(tex.format)
		if bytesPerTexel == 0 {
			return
		}
		bytesPerRow = size.Width * bytesPerTexel
	}

	rowsPerImage := layout.RowsPerImage
	if rowsPerImage == 0 {
		rowsPerImage = size.Height
	}

	depth := size.DepthOrArrayLayers
	if depth == 0 {
		depth = 1
	}

	srcBytesPerRow := uint64(bytesPerRow)
	srcRowsPerImage := uint64(rowsPerImage)
	srcBytesPerImage := srcBytesPerRow * srcRowsPerImage

	required := uint64(layout.Offset) + srcBytesPerImage*uint64(depth-1) + srcBytesPerRow*uint64(size.Height)
	if uint64(len(data)) < required {
		return
	}

	alignedBytesPerRow := alignBytesPerRow(bytesPerRow)
	dstBytesPerRow := uint64(alignedBytesPerRow)
	dstBytesPerImage := dstBytesPerRow * uint64(rowsPerImage)

	stagingSize := dstBytesPerImage * uint64(depth)
	if stagingSize == 0 {
		return
	}

	staging, err := q.device.CreateBuffer(&hal.BufferDescriptor{
		Label: "texture-write-staging",
		Size:  stagingSize,
		Usage: types.BufferUsageCopySrc | types.BufferUsageMapWrite,
	})
	if err != nil {
		return
	}
	defer q.device.DestroyBuffer(staging)

	stagingBuf, ok := staging.(*Buffer)
	if !ok || stagingBuf == nil {
		return
	}
	ptr := stagingBuf.Contents()
	if ptr == 0 {
		return
	}

	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), int(stagingSize))
	srcOffset := uint64(layout.Offset)

	for layer := uint64(0); layer < uint64(depth); layer++ {
		layerSrcBase := srcOffset + layer*srcBytesPerImage
		layerDstBase := layer * dstBytesPerImage
		for row := uint64(0); row < uint64(size.Height); row++ {
			srcIndex := layerSrcBase + row*srcBytesPerRow
			dstIndex := layerDstBase + row*dstBytesPerRow
			copy(dstBytes[dstIndex:dstIndex+srcBytesPerRow], data[srcIndex:srcIndex+srcBytesPerRow])
		}
	}

	encoder, err := q.device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{
		Label: "texture-write-encoder",
	})
	if err != nil {
		return
	}

	region := hal.BufferTextureCopy{
		BufferLayout: hal.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  alignedBytesPerRow,
			RowsPerImage: rowsPerImage,
		},
		TextureBase: hal.ImageCopyTexture{
			Texture:  dst.Texture,
			MipLevel: dst.MipLevel,
			Origin:   dst.Origin,
			Aspect:   dst.Aspect,
		},
		Size: hal.Extent3D{
			Width:              size.Width,
			Height:             size.Height,
			DepthOrArrayLayers: depth,
		},
	}

	encoder.CopyBufferToTexture(staging, dst.Texture, []hal.BufferTextureCopy{region})

	cmdBuffer, err := encoder.EndEncoding()
	if err != nil {
		return
	}
	_ = q.Submit([]hal.CommandBuffer{cmdBuffer}, nil, 0)
	if cb, ok := cmdBuffer.(*CommandBuffer); ok && cb != nil {
		_ = MsgSend(cb.raw, Sel("waitUntilCompleted"))
		cb.Destroy()
	}
}

// Present presents a surface texture to the screen.
//
// Note: On Metal, the actual presentation is scheduled via presentDrawable:
// in Submit() BEFORE the command buffer is committed. This ensures proper
// synchronization between GPU work and display.
//
// This method only releases the drawable reference. The present was already
// scheduled during Submit() if a drawable was attached to the command buffer.
func (q *Queue) Present(surface hal.Surface, texture hal.SurfaceTexture) error {
	st, ok := texture.(*SurfaceTexture)
	if !ok || st == nil {
		return nil
	}

	// Release drawable reference (presentation was scheduled in Submit)
	if st.drawable != 0 {
		Release(st.drawable)
		st.drawable = 0
	}

	return nil
}

// GetTimestampPeriod returns the timestamp period in nanoseconds.
func (q *Queue) GetTimestampPeriod() float32 {
	// Metal timestamps are in nanoseconds
	return 1.0
}

func alignBytesPerRow(value uint32) uint32 {
	const alignment = 256
	if value == 0 {
		return 0
	}
	aligned := (value + alignment - 1) &^ (alignment - 1)
	if aligned < value {
		return value
	}
	return aligned
}

func bytesPerTexel(format types.TextureFormat) uint32 {
	switch format {
	case types.TextureFormatR8Unorm,
		types.TextureFormatR8Snorm,
		types.TextureFormatR8Uint,
		types.TextureFormatR8Sint,
		types.TextureFormatStencil8:
		return 1

	case types.TextureFormatR16Uint,
		types.TextureFormatR16Sint,
		types.TextureFormatR16Float,
		types.TextureFormatRG8Unorm,
		types.TextureFormatRG8Snorm,
		types.TextureFormatRG8Uint,
		types.TextureFormatRG8Sint,
		types.TextureFormatDepth16Unorm:
		return 2

	case types.TextureFormatR32Uint,
		types.TextureFormatR32Sint,
		types.TextureFormatR32Float,
		types.TextureFormatRG16Uint,
		types.TextureFormatRG16Sint,
		types.TextureFormatRG16Float,
		types.TextureFormatRGBA8Unorm,
		types.TextureFormatRGBA8UnormSrgb,
		types.TextureFormatRGBA8Snorm,
		types.TextureFormatRGBA8Uint,
		types.TextureFormatRGBA8Sint,
		types.TextureFormatBGRA8Unorm,
		types.TextureFormatBGRA8UnormSrgb,
		types.TextureFormatRGB10A2Uint,
		types.TextureFormatRGB10A2Unorm,
		types.TextureFormatRG11B10Ufloat,
		types.TextureFormatRGB9E5Ufloat,
		types.TextureFormatDepth24Plus,
		types.TextureFormatDepth24PlusStencil8,
		types.TextureFormatDepth32Float:
		return 4

	case types.TextureFormatRG32Uint,
		types.TextureFormatRG32Sint,
		types.TextureFormatRG32Float,
		types.TextureFormatRGBA16Uint,
		types.TextureFormatRGBA16Sint,
		types.TextureFormatRGBA16Float,
		types.TextureFormatDepth32FloatStencil8:
		return 8

	case types.TextureFormatRGBA32Uint,
		types.TextureFormatRGBA32Sint,
		types.TextureFormatRGBA32Float:
		return 16
	default:
		return 0
	}
}
