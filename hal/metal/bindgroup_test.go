// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"testing"

	"github.com/gogpu/naga"
	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

func TestBindGroupLayoutMapping(t *testing.T) {
	d := &Device{}
	layout, err := d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    2,
				Visibility: types.ShaderStageFragment,
				Sampler:    &types.SamplerBindingLayout{Type: types.SamplerBindingTypeFiltering},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageVertex,
				Texture: &types.TextureBindingLayout{
					SampleType:    types.TextureSampleTypeFloat,
					ViewDimension: types.TextureViewDimension2D,
				},
			},
			{
				Binding:    0,
				Visibility: types.ShaderStageVertex,
				Buffer: &types.BufferBindingLayout{
					Type: types.BufferBindingTypeUniform,
				},
			},
			{
				Binding:    3,
				Visibility: types.ShaderStageFragment,
				Buffer: &types.BufferBindingLayout{
					Type:             types.BufferBindingTypeStorage,
					HasDynamicOffset: true,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout failed: %v", err)
	}

	mtlLayout := layout.(*BindGroupLayout)
	if mtlLayout.bufferCount != 2 || mtlLayout.textureCount != 1 || mtlLayout.samplerCount != 1 {
		t.Fatalf("unexpected binding counts: buffers=%d textures=%d samplers=%d",
			mtlLayout.bufferCount, mtlLayout.textureCount, mtlLayout.samplerCount)
	}

	info := mtlLayout.bindingInfo[0]
	if info.kind != bindingKindBuffer || info.index != 0 {
		t.Fatalf("binding 0 expected buffer index 0, got kind=%v index=%d", info.kind, info.index)
	}

	info = mtlLayout.bindingInfo[1]
	if info.kind != bindingKindTexture || info.index != 0 {
		t.Fatalf("binding 1 expected texture index 0, got kind=%v index=%d", info.kind, info.index)
	}

	info = mtlLayout.bindingInfo[2]
	if info.kind != bindingKindSampler || info.index != 0 {
		t.Fatalf("binding 2 expected sampler index 0, got kind=%v index=%d", info.kind, info.index)
	}

	info = mtlLayout.bindingInfo[3]
	if info.kind != bindingKindBuffer || info.index != 1 || !info.hasDynamicOffset {
		t.Fatalf("binding 3 expected buffer index 1 with dynamic offset, got index=%d dynamic=%v", info.index, info.hasDynamicOffset)
	}

	if idx := mtlLayout.dynamicIndex[3]; idx != 0 {
		t.Fatalf("dynamic index for binding 3 = %d, want 0", idx)
	}
}

func TestPipelineLayoutOffsets(t *testing.T) {
	d := &Device{}
	layout0, err := d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageVertex,
				Buffer:     &types.BufferBindingLayout{Type: types.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageFragment,
				Texture: &types.TextureBindingLayout{
					SampleType:    types.TextureSampleTypeFloat,
					ViewDimension: types.TextureViewDimension2D,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout[0] failed: %v", err)
	}

	layout1, err := d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageVertex,
				Buffer:     &types.BufferBindingLayout{Type: types.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageFragment,
				Buffer:     &types.BufferBindingLayout{Type: types.BufferBindingTypeStorage},
			},
			{
				Binding:    2,
				Visibility: types.ShaderStageFragment,
				Sampler:    &types.SamplerBindingLayout{Type: types.SamplerBindingTypeFiltering},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout[1] failed: %v", err)
	}

	pipelineLayout, err := d.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		BindGroupLayouts: []hal.BindGroupLayout{layout0, layout1},
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}

	mtlLayout := pipelineLayout.(*PipelineLayout)
	if got := mtlLayout.offsets[0]; got.buffer != 0 || got.texture != 0 || got.sampler != 0 {
		t.Fatalf("group 0 offsets = %+v, want all zeros", got)
	}
	if got := mtlLayout.offsets[1]; got.buffer != 1 || got.texture != 1 || got.sampler != 0 {
		t.Fatalf("group 1 offsets = %+v, want buffers=1 textures=1 samplers=0", got)
	}
}

func TestRemapModuleBindings(t *testing.T) {
	// TODO: cover storage textures and dynamic offset binding in encoder paths.
	const src = `
@group(0) @binding(0) var<uniform> u: vec4<f32>;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@group(1) @binding(0) var<storage, read> data: array<u32>;

@compute @workgroup_size(1)
fn main() {}
`

	ast, err := naga.Parse(src)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	mod, err := naga.LowerWithSource(ast, src)
	if err != nil {
		t.Fatalf("LowerWithSource failed: %v", err)
	}

	d := &Device{}
	layout0, err := d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageCompute,
				Buffer:     &types.BufferBindingLayout{Type: types.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageCompute,
				Texture: &types.TextureBindingLayout{
					SampleType:    types.TextureSampleTypeFloat,
					ViewDimension: types.TextureViewDimension2D,
				},
			},
			{
				Binding:    2,
				Visibility: types.ShaderStageCompute,
				Sampler:    &types.SamplerBindingLayout{Type: types.SamplerBindingTypeFiltering},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout[0] failed: %v", err)
	}

	layout1, err := d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageCompute,
				Buffer:     &types.BufferBindingLayout{Type: types.BufferBindingTypeStorage},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout[1] failed: %v", err)
	}

	pipelineLayout, err := d.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		BindGroupLayouts: []hal.BindGroupLayout{layout0, layout1},
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}

	if err := remapModuleBindings(mod, pipelineLayout.(*PipelineLayout)); err != nil {
		t.Fatalf("remapModuleBindings failed: %v", err)
	}

	var (
		uBinding uint32
		tBinding uint32
		sBinding uint32
		dBinding uint32
	)
	for _, gv := range mod.GlobalVariables {
		if gv.Binding == nil {
			continue
		}
		switch gv.Name {
		case "u":
			uBinding = gv.Binding.Binding
		case "t":
			tBinding = gv.Binding.Binding
		case "s":
			sBinding = gv.Binding.Binding
		case "data":
			dBinding = gv.Binding.Binding
		}
	}

	if uBinding != 0 || tBinding != 0 || sBinding != 0 {
		t.Fatalf("group 0 bindings remapped incorrectly: u=%d t=%d s=%d", uBinding, tBinding, sBinding)
	}
	if dBinding != 1 {
		t.Fatalf("group 1 buffer binding remapped to %d, want 1", dBinding)
	}
}
