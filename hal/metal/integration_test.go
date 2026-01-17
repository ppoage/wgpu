// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

const renderWGSLBasic = `
struct VSOut {
	@builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VSOut {
	var out: VSOut;
	out.position = vec4<f32>(position, 0.0, 1.0);
	return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
	return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
`

const renderWGSLTextured = `
struct VertexInput {
	@location(0) position: vec2<f32>,
	@location(1) uv: vec2<f32>,
}

struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uTint: vec4<f32>;
@group(0) @binding(1) var uSampler: sampler;
@group(0) @binding(2) var uTexture: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VSOut {
	var out: VSOut;
	out.position = vec4<f32>(input.position, 0.0, 1.0);
	out.uv = input.uv;
	return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
	return textureSample(uTexture, uSampler, input.uv) * uTint;
}
`

const computeWGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i < 4u) {
		data[i] = i * 3u + 1u;
	}
}
`

func float32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}

func uint16Bytes(values []uint16) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*2)
}

func submitAndWait(t *testing.T, queue *Queue, cmd hal.CommandBuffer) {
	t.Helper()
	if err := queue.Submit([]hal.CommandBuffer{cmd}, nil, 0); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	if cb, ok := cmd.(*CommandBuffer); ok && cb != nil {
		_ = MsgSend(cb.raw, Sel("waitUntilCompleted"))
		cb.Destroy()
	}
}

func readbackTexture(t *testing.T, device *Device, queue *Queue, tex hal.Texture, width, height uint32, format types.TextureFormat) ([]byte, uint32) {
	t.Helper()

	bpp := bytesPerTexel(format)
	if bpp == 0 {
		t.Fatalf("bytesPerTexel returned 0 for %v", format)
	}

	rowBytes := width * bpp
	alignedRowBytes := alignBytesPerRow(rowBytes)
	readbackSize := uint64(alignedRowBytes) * uint64(height)

	buf, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "render-readback",
		Size:  readbackSize,
		Usage: types.BufferUsageCopyDst | types.BufferUsageMapRead,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(buf)

	encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "render-readback-encoder"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder failed: %v", err)
	}

	encoder.CopyTextureToBuffer(tex, buf, []hal.BufferTextureCopy{
		{
			BufferLayout: hal.ImageDataLayout{
				Offset:       0,
				BytesPerRow:  alignedRowBytes,
				RowsPerImage: height,
			},
			TextureBase: hal.ImageCopyTexture{
				Texture:  tex,
				MipLevel: 0,
				Origin:   hal.Origin3D{X: 0, Y: 0, Z: 0},
				Aspect:   types.TextureAspectAll,
			},
			Size: hal.Extent3D{
				Width:              width,
				Height:             height,
				DepthOrArrayLayers: 1,
			},
		},
	})

	cmd, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	submitAndWait(t, queue, cmd)

	mtlBuf := buf.(*Buffer)
	ptr := mtlBuf.Contents()
	if ptr == 0 {
		t.Fatal("Buffer contents pointer is nil")
	}
	readback := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), int(readbackSize))
	copyOut := append([]byte(nil), readback...)
	return copyOut, alignedRowBytes
}

func colorDistance(a, b byte) float64 {
	return math.Abs(float64(a) - float64(b))
}

func TestRenderPassTriangleReadback(t *testing.T) {
	device, queue, cleanup := openTestDevice(t)
	defer cleanup()

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "triangle-module",
		Source: hal.ShaderSource{
			WGSL: renderWGSLBasic,
		},
	})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer device.DestroyShaderModule(module)

	layout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label:            "triangle-layout",
		BindGroupLayouts: nil,
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	defer device.DestroyPipelineLayout(layout)

	pipeline, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Label:  "triangle-pipeline",
		Layout: layout,
		Vertex: hal.VertexState{
			Module:     module,
			EntryPoint: "vs_main",
			Buffers: []types.VertexBufferLayout{
				{
					ArrayStride: 8,
					StepMode:    types.VertexStepModeVertex,
					Attributes: []types.VertexAttribute{
						{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					},
				},
			},
		},
		Primitive: types.PrimitiveState{
			Topology: types.PrimitiveTopologyTriangleList,
		},
		Multisample: types.MultisampleState{
			Count: 1,
		},
		Fragment: &hal.FragmentState{
			Module:     module,
			EntryPoint: "fs_main",
			Targets: []types.ColorTargetState{
				{Format: types.TextureFormatBGRA8Unorm, WriteMask: types.ColorWriteMaskAll},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline failed: %v", err)
	}
	defer device.DestroyRenderPipeline(pipeline)

	vertices := []float32{
		-1.0, -1.0,
		1.0, -1.0,
		0.0, 1.0,
	}
	vbo, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "triangle-vertices",
		Size:  uint64(len(vertices)) * 4,
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(vbo)
	queue.WriteBuffer(vbo, 0, float32Bytes(vertices))

	size := hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}
	target, err := device.CreateTexture(&hal.TextureDescriptor{
		Label:         "triangle-target",
		Size:          size,
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatBGRA8Unorm,
		Usage:         types.TextureUsageRenderAttachment | types.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	defer device.DestroyTexture(target)

	view, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer device.DestroyTextureView(view)

	encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "triangle-encoder"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder failed: %v", err)
	}

	pass := encoder.BeginRenderPass(&hal.RenderPassDescriptor{
		ColorAttachments: []hal.RenderPassColorAttachment{
			{
				View:       view,
				LoadOp:     types.LoadOpClear,
				StoreOp:    types.StoreOpStore,
				ClearValue: types.ColorBlack,
			},
		},
	})
	pass.SetPipeline(pipeline)
	pass.SetViewport(0, 0, float32(size.Width), float32(size.Height), 0, 1)
	pass.SetScissorRect(0, 0, size.Width, size.Height)
	pass.SetVertexBuffer(0, vbo, 0)
	pass.Draw(3, 1, 0, 0)
	pass.End()

	cmd, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	submitAndWait(t, queue, cmd)

	readback, rowStride := readbackTexture(t, device, queue, target, size.Width, size.Height, types.TextureFormatBGRA8Unorm)
	x, y := 4, 4
	idx := int(rowStride)*y + x*4
	if idx+3 >= len(readback) {
		t.Fatalf("readback index out of range: %d", idx)
	}
	got := [4]byte{readback[idx], readback[idx+1], readback[idx+2], readback[idx+3]}
	want := [4]byte{0xFF, 0x00, 0x00, 0xFF} // blue in BGRA
	for i := range want {
		if colorDistance(got[i], want[i]) > 2 {
			t.Fatalf("pixel mismatch: got %v want %v", got, want)
		}
	}
}

func TestRenderPassTexturedQuadReadback(t *testing.T) {
	device, queue, cleanup := openTestDevice(t)
	defer cleanup()

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "textured-module",
		Source: hal.ShaderSource{
			WGSL: renderWGSLTextured,
		},
	})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer device.DestroyShaderModule(module)

	bgl, err := device.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Label: "textured-bgl",
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageFragment,
				Buffer: &types.BufferBindingLayout{
					Type:             types.BufferBindingTypeUniform,
					HasDynamicOffset: false,
					MinBindingSize:   16,
				},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageFragment,
				Sampler: &types.SamplerBindingLayout{
					Type: types.SamplerBindingTypeFiltering,
				},
			},
			{
				Binding:    2,
				Visibility: types.ShaderStageFragment,
				Texture: &types.TextureBindingLayout{
					SampleType:    types.TextureSampleTypeFloat,
					ViewDimension: types.TextureViewDimension2D,
					Multisampled:  false,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout failed: %v", err)
	}
	defer device.DestroyBindGroupLayout(bgl)

	layout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label:            "textured-layout",
		BindGroupLayouts: []hal.BindGroupLayout{bgl},
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	defer device.DestroyPipelineLayout(layout)

	pipeline, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Label:  "textured-pipeline",
		Layout: layout,
		Vertex: hal.VertexState{
			Module:     module,
			EntryPoint: "vs_main",
			Buffers: []types.VertexBufferLayout{
				{
					ArrayStride: 16,
					StepMode:    types.VertexStepModeVertex,
					Attributes: []types.VertexAttribute{
						{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
						{Format: types.VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
					},
				},
			},
		},
		Primitive: types.PrimitiveState{
			Topology: types.PrimitiveTopologyTriangleList,
		},
		Multisample: types.MultisampleState{
			Count: 1,
		},
		Fragment: &hal.FragmentState{
			Module:     module,
			EntryPoint: "fs_main",
			Targets: []types.ColorTargetState{
				{Format: types.TextureFormatBGRA8Unorm, WriteMask: types.ColorWriteMaskAll},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline failed: %v", err)
	}
	defer device.DestroyRenderPipeline(pipeline)

	vertices := []float32{
		-1.0, -1.0, 0.0, 0.0,
		1.0, -1.0, 1.0, 0.0,
		1.0, 1.0, 1.0, 1.0,
		-1.0, 1.0, 0.0, 1.0,
	}
	indices := []uint16{0, 1, 2, 2, 3, 0}

	vbo, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "quad-vertices",
		Size:  uint64(len(vertices)) * 4,
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(vbo)
	queue.WriteBuffer(vbo, 0, float32Bytes(vertices))

	ibo, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "quad-indices",
		Size:  uint64(len(indices)) * 2,
		Usage: types.BufferUsageIndex | types.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(ibo)
	queue.WriteBuffer(ibo, 0, uint16Bytes(indices))

	tint := []float32{1.0, 1.0, 1.0, 1.0}
	ubo, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "tint-uniform",
		Size:  uint64(len(tint)) * 4,
		Usage: types.BufferUsageUniform | types.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(ubo)
	queue.WriteBuffer(ubo, 0, float32Bytes(tint))

	texSize := hal.Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1}
	texture, err := device.CreateTexture(&hal.TextureDescriptor{
		Label:         "sample-texture",
		Size:          texSize,
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatRGBA8Unorm,
		Usage:         types.TextureUsageCopyDst | types.TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	defer device.DestroyTexture(texture)

	texData := []byte{0x00, 0xFF, 0x00, 0xFF}
	queue.WriteTexture(
		&hal.ImageCopyTexture{
			Texture:  texture,
			MipLevel: 0,
			Origin:   hal.Origin3D{X: 0, Y: 0, Z: 0},
			Aspect:   types.TextureAspectAll,
		},
		texData,
		&hal.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  4,
			RowsPerImage: 1,
		},
		&texSize,
	)

	texView, err := device.CreateTextureView(texture, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer device.DestroyTextureView(texView)

	sampler, err := device.CreateSampler(&hal.SamplerDescriptor{
		Label:         "nearest-sampler",
		AddressModeU:  types.AddressModeClampToEdge,
		AddressModeV:  types.AddressModeClampToEdge,
		AddressModeW:  types.AddressModeClampToEdge,
		MagFilter:     types.FilterModeNearest,
		MinFilter:     types.FilterModeNearest,
		MipmapFilter:  types.FilterModeNearest,
		LodMinClamp:   0,
		LodMaxClamp:   1,
		Compare:       types.CompareFunctionUndefined,
		Anisotropy:    1,
	})
	if err != nil {
		t.Fatalf("CreateSampler failed: %v", err)
	}
	defer device.DestroySampler(sampler)

	bindGroup, err := device.CreateBindGroup(&hal.BindGroupDescriptor{
		Label:  "textured-bind-group",
		Layout: bgl,
		Entries: []types.BindGroupEntry{
			{
				Binding:  0,
				Resource: types.BufferBinding{Buffer: types.BufferHandle(ubo.(*Buffer).raw), Offset: 0, Size: 16},
			},
			{
				Binding:  1,
				Resource: types.SamplerBinding{Sampler: types.SamplerHandle(sampler.(*Sampler).raw)},
			},
			{
				Binding:  2,
				Resource: types.TextureViewBinding{TextureView: types.TextureViewHandle(texView.(*TextureView).raw)},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroup failed: %v", err)
	}
	defer device.DestroyBindGroup(bindGroup)

	targetSize := hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}
	target, err := device.CreateTexture(&hal.TextureDescriptor{
		Label:         "textured-target",
		Size:          targetSize,
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatBGRA8Unorm,
		Usage:         types.TextureUsageRenderAttachment | types.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	defer device.DestroyTexture(target)

	targetView, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer device.DestroyTextureView(targetView)

	encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "textured-encoder"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder failed: %v", err)
	}

	pass := encoder.BeginRenderPass(&hal.RenderPassDescriptor{
		ColorAttachments: []hal.RenderPassColorAttachment{
			{
				View:       targetView,
				LoadOp:     types.LoadOpClear,
				StoreOp:    types.StoreOpStore,
				ClearValue: types.ColorBlack,
			},
		},
	})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.SetVertexBuffer(0, vbo, 0)
	pass.SetIndexBuffer(ibo, types.IndexFormatUint16, 0)
	pass.DrawIndexed(uint32(len(indices)), 1, 0, 0, 0)
	pass.End()

	cmd, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	submitAndWait(t, queue, cmd)

	readback, rowStride := readbackTexture(t, device, queue, target, targetSize.Width, targetSize.Height, types.TextureFormatBGRA8Unorm)
	x, y := 1, 1
	idx := int(rowStride)*y + x*4
	got := [4]byte{readback[idx], readback[idx+1], readback[idx+2], readback[idx+3]}
	want := [4]byte{0x00, 0xFF, 0x00, 0xFF}
	for i := range want {
		if colorDistance(got[i], want[i]) > 2 {
			t.Fatalf("pixel mismatch: got %v want %v", got, want)
		}
	}
}

func TestComputePipelineDispatchWritesBuffer(t *testing.T) {
	device, queue, cleanup := openTestDevice(t)
	defer cleanup()

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "compute-module",
		Source: hal.ShaderSource{
			WGSL: computeWGSL,
		},
	})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer device.DestroyShaderModule(module)

	bgl, err := device.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Label: "compute-bgl",
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageCompute,
				Buffer: &types.BufferBindingLayout{
					Type:             types.BufferBindingTypeStorage,
					HasDynamicOffset: false,
					MinBindingSize:   16,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout failed: %v", err)
	}
	defer device.DestroyBindGroupLayout(bgl)

	layout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label:            "compute-layout",
		BindGroupLayouts: []hal.BindGroupLayout{bgl},
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	defer device.DestroyPipelineLayout(layout)

	pipeline, err := device.CreateComputePipeline(&hal.ComputePipelineDescriptor{
		Label:  "compute-pipeline",
		Layout: layout,
		Compute: hal.ComputeState{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		t.Fatalf("CreateComputePipeline failed: %v", err)
	}
	mtlPipeline, ok := pipeline.(*ComputePipeline)
	if !ok || mtlPipeline == nil {
		t.Fatal("CreateComputePipeline returned nil or non-metal pipeline")
	}
	if mtlPipeline.raw == 0 {
		t.Fatal("ComputePipeline raw handle is nil")
	}
	if mtlPipeline.layout == nil || mtlPipeline.layout != layout {
		t.Fatal("ComputePipeline layout mismatch or nil")
	}
	if mtlPipeline.workgroupSize.Width == 0 || mtlPipeline.workgroupSize.Height == 0 || mtlPipeline.workgroupSize.Depth == 0 {
		t.Fatalf("ComputePipeline workgroup size invalid: %+v", mtlPipeline.workgroupSize)
	}
	defer device.DestroyComputePipeline(pipeline)

	buffer, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "compute-buffer",
		Size:  16,
		Usage: types.BufferUsageStorage | types.BufferUsageMapRead,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	mtlBuffer, ok := buffer.(*Buffer)
	if !ok || mtlBuffer == nil {
		t.Fatal("CreateBuffer returned nil or non-metal buffer")
	}
	if mtlBuffer.raw == 0 {
		t.Fatal("Buffer raw handle is nil")
	}
	if mtlBuffer.size != 16 {
		t.Fatalf("Buffer size mismatch: got %d want 16", mtlBuffer.size)
	}
	defer device.DestroyBuffer(buffer)

	bindGroup, err := device.CreateBindGroup(&hal.BindGroupDescriptor{
		Label:  "compute-bind-group",
		Layout: bgl,
		Entries: []types.BindGroupEntry{
			{
				Binding:  0,
				Resource: types.BufferBinding{Buffer: types.BufferHandle(mtlBuffer.raw), Offset: 0, Size: 16},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroup failed: %v", err)
	}
	mtlBindGroup, ok := bindGroup.(*BindGroup)
	if !ok || mtlBindGroup == nil {
		t.Fatal("CreateBindGroup returned nil or non-metal bind group")
	}
	if mtlBindGroup.layout == nil || mtlBindGroup.layout != bgl {
		t.Fatal("BindGroup layout mismatch or nil")
	}
	defer device.DestroyBindGroup(bindGroup)

	encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "compute-encoder"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder failed: %v", err)
	}

	pass := encoder.BeginComputePass(&hal.ComputePassDescriptor{})
	if pass == nil {
		t.Fatal("BeginComputePass returned nil")
	}
	mtlPass, ok := pass.(*ComputePassEncoder)
	if !ok || mtlPass == nil {
		t.Fatal("BeginComputePass returned non-metal encoder")
	}
	if mtlPass.raw == 0 {
		t.Fatal("ComputePassEncoder raw handle is nil")
	}
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	fmt.Fprintf(os.Stderr, "compute dispatch: passRaw=%#x threadgroups=%+v threadsPerThreadgroup=%+v mtlSizeType=%+v\n",
		mtlPass.raw,
		MTLSize{Width: 4, Height: 1, Depth: 1},
		mtlPipeline.workgroupSize,
		mtlSizeType,
	)
	pass.Dispatch(4, 1, 1)
	pass.End()

	cmd, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	submitAndWait(t, queue, cmd)

	ptr := mtlBuffer.Contents()
	if ptr == 0 {
		t.Fatal("Buffer contents pointer is nil")
	}
	t.Logf("compute buffer ptr=%#x workgroup=%+v offsets=%+v",
		ptr, mtlPipeline.workgroupSize, mtlPipeline.layout.offsets)
	raw := unsafe.Slice((*uint32)(unsafe.Pointer(ptr)), 4)
	want := []uint32{1, 4, 7, 10}
	for i, v := range want {
		if raw[i] != v {
			t.Fatalf("buffer[%d] = %d, want %d (raw=%v)", i, raw[i], v, raw)
		}
	}
}

func TestRenderPassBlendConstantReadback(t *testing.T) {
	device, queue, cleanup := openTestDevice(t)
	defer cleanup()

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "blend-module",
		Source: hal.ShaderSource{
			WGSL: renderWGSLBasic,
		},
	})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer device.DestroyShaderModule(module)

	layout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label:            "blend-layout",
		BindGroupLayouts: nil,
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	defer device.DestroyPipelineLayout(layout)

	blend := &types.BlendState{
		Color: types.BlendComponent{
			SrcFactor: types.BlendFactorConstant,
			DstFactor: types.BlendFactorOneMinusConstant,
			Operation: types.BlendOperationAdd,
		},
		Alpha: types.BlendComponent{
			SrcFactor: types.BlendFactorOne,
			DstFactor: types.BlendFactorZero,
			Operation: types.BlendOperationAdd,
		},
	}

	pipeline, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Label:  "blend-pipeline",
		Layout: layout,
		Vertex: hal.VertexState{
			Module:     module,
			EntryPoint: "vs_main",
			Buffers: []types.VertexBufferLayout{
				{
					ArrayStride: 8,
					StepMode:    types.VertexStepModeVertex,
					Attributes: []types.VertexAttribute{
						{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					},
				},
			},
		},
		Primitive: types.PrimitiveState{
			Topology: types.PrimitiveTopologyTriangleList,
		},
		Multisample: types.MultisampleState{
			Count: 1,
		},
		Fragment: &hal.FragmentState{
			Module:     module,
			EntryPoint: "fs_main",
			Targets: []types.ColorTargetState{
				{Format: types.TextureFormatBGRA8Unorm, Blend: blend, WriteMask: types.ColorWriteMaskAll},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline failed: %v", err)
	}
	defer device.DestroyRenderPipeline(pipeline)

	vertices := []float32{
		-1.0, -1.0,
		1.0, -1.0,
		0.0, 1.0,
	}
	vbo, err := device.CreateBuffer(&hal.BufferDescriptor{
		Label: "blend-vertices",
		Size:  uint64(len(vertices)) * 4,
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateBuffer failed: %v", err)
	}
	defer device.DestroyBuffer(vbo)
	queue.WriteBuffer(vbo, 0, float32Bytes(vertices))

	size := hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}
	target, err := device.CreateTexture(&hal.TextureDescriptor{
		Label:         "blend-target",
		Size:          size,
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatBGRA8Unorm,
		Usage:         types.TextureUsageRenderAttachment | types.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	defer device.DestroyTexture(target)

	view, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer device.DestroyTextureView(view)

	encoder, err := device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "blend-encoder"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder failed: %v", err)
	}

	pass := encoder.BeginRenderPass(&hal.RenderPassDescriptor{
		ColorAttachments: []hal.RenderPassColorAttachment{
			{
				View:       view,
				LoadOp:     types.LoadOpClear,
				StoreOp:    types.StoreOpStore,
				ClearValue: types.ColorBlack,
			},
		},
	})
	pass.SetPipeline(pipeline)
	pass.SetBlendConstant(&types.Color{R: 0.5, G: 0.5, B: 0.5, A: 1.0})
	pass.SetVertexBuffer(0, vbo, 0)
	pass.Draw(3, 1, 0, 0)
	pass.End()

	cmd, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	submitAndWait(t, queue, cmd)

	readback, rowStride := readbackTexture(t, device, queue, target, size.Width, size.Height, types.TextureFormatBGRA8Unorm)
	idx := int(rowStride)*1 + 1*4
	got := readback[idx] // blue channel in BGRA
	if colorDistance(got, 0x80) > 8 {
		t.Fatalf("blend output mismatch: got %d want ~128", got)
	}
}
