// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"testing"

	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

const pipelineWGSLBasic = `
struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VSOut {
	var out: VSOut;
	out.position = vec4<f32>(position, 0.0, 1.0);
	out.color = vec3<f32>(1.0, 0.0, 0.0);
	return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
	return vec4<f32>(input.color, 1.0);
}
`

const pipelineWGSLTextured = `
struct VertexInput {
	@location(0) position: vec2<f32>,
	@location(1) color: vec3<f32>,
}

struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uTint: vec4<f32>;
@group(0) @binding(1) var uSampler: sampler;
@group(0) @binding(2) var uTexture: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VSOut {
	var out: VSOut;
	out.position = vec4<f32>(input.position, 0.0, 1.0);
	out.color = input.color;
	return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
	let texColor = textureSample(uTexture, uSampler, vec2<f32>(0.5, 0.5));
	return vec4<f32>(input.color, 1.0) * uTint * texColor;
}
`

func TestRenderPipelineCreation(t *testing.T) {
	device, _, cleanup := openTestDevice(t)
	defer cleanup()

	tests := []struct {
		name       string
		wgsl       string
		withLayout bool
	}{
		{name: "BasicNoBindings", wgsl: pipelineWGSLBasic, withLayout: false},
		{name: "TexturedBindings", wgsl: pipelineWGSLTextured, withLayout: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
				Label: "pipeline-test",
				Source: hal.ShaderSource{
					WGSL: tc.wgsl,
				},
			})
			if err != nil {
				t.Fatalf("CreateShaderModule failed: %v", err)
			}
			defer device.DestroyShaderModule(module)

			var layout hal.PipelineLayout
			if tc.withLayout {
				bgl, err := device.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
					Label: "pipeline-test-layout",
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

				layout, err = device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
					Label:            "pipeline-test-pipeline-layout",
					BindGroupLayouts: []hal.BindGroupLayout{bgl},
				})
				if err != nil {
					t.Fatalf("CreatePipelineLayout failed: %v", err)
				}
				defer device.DestroyPipelineLayout(layout)
			} else {
				var err error
				layout, err = device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
					Label:            "pipeline-test-empty-layout",
					BindGroupLayouts: nil,
				})
				if err != nil {
					t.Fatalf("CreatePipelineLayout failed: %v", err)
				}
				defer device.DestroyPipelineLayout(layout)
			}

			vertexBuffers := []types.VertexBufferLayout{
				{
					ArrayStride: 20,
					StepMode:    types.VertexStepModeVertex,
					Attributes: []types.VertexAttribute{
						{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
						{Format: types.VertexFormatFloat32x3, Offset: 8, ShaderLocation: 1},
					},
				},
			}
			if tc.name == "BasicNoBindings" {
				vertexBuffers = []types.VertexBufferLayout{
					{
						ArrayStride: 8,
						StepMode:    types.VertexStepModeVertex,
						Attributes: []types.VertexAttribute{
							{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
						},
					},
				}
			}

			pipeline, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
				Label:  "pipeline-test",
				Layout: layout,
				Vertex: hal.VertexState{
					Module:     module,
					EntryPoint: "vs_main",
					Buffers:    vertexBuffers,
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
						{Format: types.TextureFormatBGRA8Unorm},
					},
				},
			})

			if err != nil {
				t.Fatalf("CreateRenderPipeline failed: %v", err)
			}
			device.DestroyRenderPipeline(pipeline)
		})
	}
}

func TestRenderPipelineBlendState(t *testing.T) {
	device, _, cleanup := openTestDevice(t)
	defer cleanup()

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "pipeline-blend-test",
		Source: hal.ShaderSource{
			WGSL: pipelineWGSLBasic,
		},
	})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer device.DestroyShaderModule(module)

	layout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label:            "pipeline-blend-layout",
		BindGroupLayouts: nil,
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	defer device.DestroyPipelineLayout(layout)

	blend := &types.BlendState{
		Color: types.BlendComponent{
			SrcFactor: types.BlendFactorSrcAlpha,
			DstFactor: types.BlendFactorOneMinusSrcAlpha,
			Operation: types.BlendOperationAdd,
		},
		Alpha: types.BlendComponent{
			SrcFactor: types.BlendFactorOne,
			DstFactor: types.BlendFactorOneMinusSrcAlpha,
			Operation: types.BlendOperationAdd,
		},
	}

	pipeline, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Label:  "pipeline-blend-test",
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
				{
					Format:    types.TextureFormatBGRA8Unorm,
					Blend:     blend,
					WriteMask: types.ColorWriteMaskRed | types.ColorWriteMaskGreen,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline (blend) failed: %v", err)
	}
	device.DestroyRenderPipeline(pipeline)
}
