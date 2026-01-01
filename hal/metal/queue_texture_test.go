// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

func openTestDevice(t *testing.T) (*Device, *Queue, func()) {
	t.Helper()

	if err := Init(); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	backend := Backend{}
	inst, err := backend.CreateInstance(&hal.InstanceDescriptor{
		Backends: types.Backends(1 << types.BackendMetal),
	})
	if err != nil {
		t.Fatalf("CreateInstance failed: %v", err)
	}
	instance := inst.(*Instance)

	adapters := instance.EnumerateAdapters(nil)
	if len(adapters) == 0 {
		t.Skip("no Metal adapters available")
	}

	adapter := adapters[0].Adapter
	open, err := adapter.Open(types.Features(0), types.DefaultLimits())
	if err != nil {
		t.Fatalf("Adapter.Open failed: %v", err)
	}

	device := open.Device.(*Device)
	queue := open.Queue.(*Queue)

	cleanup := func() {
		open.Device.Destroy()
		adapter.Destroy()
	}
	return device, queue, cleanup
}

func TestQueueWriteTextureRoundTrip(t *testing.T) {
	device, queue, cleanup := openTestDevice(t)
	defer cleanup()

	type formatCase struct {
		name   string
		format types.TextureFormat
		bpp    uint32
	}

	formats := []formatCase{
		{name: "R8Unorm", format: types.TextureFormatR8Unorm, bpp: 1},
		{name: "RG8Unorm", format: types.TextureFormatRG8Unorm, bpp: 2},
		{name: "RGBA8Unorm", format: types.TextureFormatRGBA8Unorm, bpp: 4},
		{name: "RGBA16Float", format: types.TextureFormatRGBA16Float, bpp: 8},
	}

	for _, tc := range formats {
		t.Run(tc.name, func(t *testing.T) {
			size := hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}
			tex, err := device.CreateTexture(&hal.TextureDescriptor{
				Label:         "write-texture-test",
				Size:          size,
				MipLevelCount: 1,
				SampleCount:   1,
				Dimension:     types.TextureDimension2D,
				Format:        tc.format,
				Usage:         types.TextureUsageCopyDst | types.TextureUsageCopySrc,
			})
			if err != nil {
				t.Skipf("format not supported: %v", err)
			}
			defer device.DestroyTexture(tex)

			rowBytes := size.Width * tc.bpp
			data := make([]byte, rowBytes*size.Height)
			for i := range data {
				data[i] = byte((i * 17) & 0xFF)
			}

			queue.WriteTexture(
				&hal.ImageCopyTexture{
					Texture:  tex,
					MipLevel: 0,
					Origin:   hal.Origin3D{X: 0, Y: 0, Z: 0},
					Aspect:   types.TextureAspectAll,
				},
				data,
				&hal.ImageDataLayout{
					Offset:       0,
					BytesPerRow:  uint32(rowBytes),
					RowsPerImage: size.Height,
				},
				&size,
			)

			alignedRowBytes := alignBytesPerRow(uint32(rowBytes))
			readbackSize := uint64(alignedRowBytes) * uint64(size.Height)
			buf, err := device.CreateBuffer(&hal.BufferDescriptor{
				Label: "texture-readback",
				Size:  readbackSize,
				Usage: types.BufferUsageCopyDst | types.BufferUsageMapRead,
			})
			if err != nil {
				t.Fatalf("CreateBuffer failed: %v", err)
			}
			defer device.DestroyBuffer(buf)

			encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "readback-encoder"})
			if err != nil {
				t.Fatalf("CreateCommandEncoder failed: %v", err)
			}

			encoder.CopyTextureToBuffer(tex, buf, []hal.BufferTextureCopy{
				{
					BufferLayout: hal.ImageDataLayout{
						Offset:       0,
						BytesPerRow:  alignedRowBytes,
						RowsPerImage: size.Height,
					},
					TextureBase: hal.ImageCopyTexture{
						Texture:  tex,
						MipLevel: 0,
						Origin:   hal.Origin3D{X: 0, Y: 0, Z: 0},
						Aspect:   types.TextureAspectAll,
					},
					Size: size,
				},
			})

			cmdBuffer, err := encoder.EndEncoding()
			if err != nil {
				t.Fatalf("EndEncoding failed: %v", err)
			}

			if err := queue.Submit([]hal.CommandBuffer{cmdBuffer}, nil, 0); err != nil {
				t.Fatalf("Submit failed: %v", err)
			}

			if cb, ok := cmdBuffer.(*CommandBuffer); ok && cb != nil {
				_ = MsgSend(cb.raw, Sel("waitUntilCompleted"))
				cb.Destroy()
			}

			mtlBuf := buf.(*Buffer)
			ptr := mtlBuf.Contents()
			if ptr == 0 {
				t.Fatal("Buffer contents pointer is nil")
			}

			readback := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), int(readbackSize))
			rowSize := int(rowBytes)
			aligned := int(alignedRowBytes)
			for y := 0; y < int(size.Height); y++ {
				got := readback[y*aligned : y*aligned+rowSize]
				want := data[y*rowSize : (y+1)*rowSize]
				if !bytes.Equal(got, want) {
					t.Fatalf("row %d mismatch for %s", y, tc.name)
				}
			}
		})
	}
}
