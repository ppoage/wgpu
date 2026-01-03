// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"testing"

	"github.com/gogpu/wgpu/types"
)

func TestVertexFormatToMTL(t *testing.T) {
	cases := []struct {
		format types.VertexFormat
		want   MTLVertexFormat
	}{
		{types.VertexFormatUint8x2, MTLVertexFormatUChar2},
		{types.VertexFormatUint8x4, MTLVertexFormatUChar4},
		{types.VertexFormatSint8x2, MTLVertexFormatChar2},
		{types.VertexFormatSint8x4, MTLVertexFormatChar4},
		{types.VertexFormatUnorm8x2, MTLVertexFormatUChar2Normalized},
		{types.VertexFormatUnorm8x4, MTLVertexFormatUChar4Normalized},
		{types.VertexFormatSnorm8x2, MTLVertexFormatChar2Normalized},
		{types.VertexFormatSnorm8x4, MTLVertexFormatChar4Normalized},
		{types.VertexFormatUint16x2, MTLVertexFormatUShort2},
		{types.VertexFormatUint16x4, MTLVertexFormatUShort4},
		{types.VertexFormatSint16x2, MTLVertexFormatShort2},
		{types.VertexFormatSint16x4, MTLVertexFormatShort4},
		{types.VertexFormatUnorm16x2, MTLVertexFormatUShort2Normalized},
		{types.VertexFormatUnorm16x4, MTLVertexFormatUShort4Normalized},
		{types.VertexFormatSnorm16x2, MTLVertexFormatShort2Normalized},
		{types.VertexFormatSnorm16x4, MTLVertexFormatShort4Normalized},
		{types.VertexFormatFloat16x2, MTLVertexFormatHalf2},
		{types.VertexFormatFloat16x4, MTLVertexFormatHalf4},
		{types.VertexFormatFloat32, MTLVertexFormatFloat},
		{types.VertexFormatFloat32x2, MTLVertexFormatFloat2},
		{types.VertexFormatFloat32x3, MTLVertexFormatFloat3},
		{types.VertexFormatFloat32x4, MTLVertexFormatFloat4},
		{types.VertexFormatUint32, MTLVertexFormatUInt},
		{types.VertexFormatUint32x2, MTLVertexFormatUInt2},
		{types.VertexFormatUint32x3, MTLVertexFormatUInt3},
		{types.VertexFormatUint32x4, MTLVertexFormatUInt4},
		{types.VertexFormatSint32, MTLVertexFormatInt},
		{types.VertexFormatSint32x2, MTLVertexFormatInt2},
		{types.VertexFormatSint32x3, MTLVertexFormatInt3},
		{types.VertexFormatSint32x4, MTLVertexFormatInt4},
		{types.VertexFormatUnorm1010102, MTLVertexFormatUInt1010102Normalized},
	}

	for _, c := range cases {
		got, ok := vertexFormatToMTL(c.format)
		if !ok {
			t.Fatalf("vertexFormatToMTL(%v) ok=false", c.format)
		}
		if got != c.want {
			t.Fatalf("vertexFormatToMTL(%v)=%v, want %v", c.format, got, c.want)
		}
	}

	if _, ok := vertexFormatToMTL(types.VertexFormat(0xff)); ok {
		t.Fatal("vertexFormatToMTL(invalid) ok=true, want false")
	}
}

func TestVertexStepModeToMTL(t *testing.T) {
	if got, ok := vertexStepModeToMTL(types.VertexStepModeVertex); !ok || got != MTLVertexStepFunctionPerVertex {
		t.Fatalf("vertexStepModeToMTL(Vertex)=%v, ok=%v", got, ok)
	}
	if got, ok := vertexStepModeToMTL(types.VertexStepModeInstance); !ok || got != MTLVertexStepFunctionPerInstance {
		t.Fatalf("vertexStepModeToMTL(Instance)=%v, ok=%v", got, ok)
	}
	if _, ok := vertexStepModeToMTL(types.VertexStepMode(99)); ok {
		t.Fatal("vertexStepModeToMTL(invalid) ok=true, want false")
	}
}

func TestBlendFactorToMTL(t *testing.T) {
	cases := []struct {
		factor types.BlendFactor
		want   MTLBlendFactor
	}{
		{types.BlendFactorZero, MTLBlendFactorZero},
		{types.BlendFactorOne, MTLBlendFactorOne},
		{types.BlendFactorSrc, MTLBlendFactorSourceColor},
		{types.BlendFactorOneMinusSrc, MTLBlendFactorOneMinusSourceColor},
		{types.BlendFactorSrcAlpha, MTLBlendFactorSourceAlpha},
		{types.BlendFactorOneMinusSrcAlpha, MTLBlendFactorOneMinusSourceAlpha},
		{types.BlendFactorDst, MTLBlendFactorDestinationColor},
		{types.BlendFactorOneMinusDst, MTLBlendFactorOneMinusDestinationColor},
		{types.BlendFactorDstAlpha, MTLBlendFactorDestinationAlpha},
		{types.BlendFactorOneMinusDstAlpha, MTLBlendFactorOneMinusDestinationAlpha},
		{types.BlendFactorSrcAlphaSaturated, MTLBlendFactorSourceAlphaSaturated},
		{types.BlendFactorConstant, MTLBlendFactorBlendColor},
		{types.BlendFactorOneMinusConstant, MTLBlendFactorOneMinusBlendColor},
	}

	for _, c := range cases {
		if got := blendFactorToMTL(c.factor); got != c.want {
			t.Fatalf("blendFactorToMTL(%v)=%v, want %v", c.factor, got, c.want)
		}
	}

	if got := blendFactorToMTL(types.BlendFactor(0xff)); got != MTLBlendFactorOne {
		t.Fatalf("blendFactorToMTL(invalid)=%v, want %v", got, MTLBlendFactorOne)
	}
}

func TestBlendOperationToMTL(t *testing.T) {
	cases := []struct {
		op   types.BlendOperation
		want MTLBlendOperation
	}{
		{types.BlendOperationAdd, MTLBlendOperationAdd},
		{types.BlendOperationSubtract, MTLBlendOperationSubtract},
		{types.BlendOperationReverseSubtract, MTLBlendOperationReverseSubtract},
		{types.BlendOperationMin, MTLBlendOperationMin},
		{types.BlendOperationMax, MTLBlendOperationMax},
	}

	for _, c := range cases {
		if got := blendOperationToMTL(c.op); got != c.want {
			t.Fatalf("blendOperationToMTL(%v)=%v, want %v", c.op, got, c.want)
		}
	}

	if got := blendOperationToMTL(types.BlendOperation(0xff)); got != MTLBlendOperationAdd {
		t.Fatalf("blendOperationToMTL(invalid)=%v, want %v", got, MTLBlendOperationAdd)
	}
}

func TestColorWriteMaskToMTL(t *testing.T) {
	cases := []struct {
		mask types.ColorWriteMask
		want MTLColorWriteMask
	}{
		{0, 0},
		{types.ColorWriteMaskRed, MTLColorWriteMaskRed},
		{types.ColorWriteMaskGreen, MTLColorWriteMaskGreen},
		{types.ColorWriteMaskBlue, MTLColorWriteMaskBlue},
		{types.ColorWriteMaskAlpha, MTLColorWriteMaskAlpha},
		{types.ColorWriteMaskRed | types.ColorWriteMaskGreen, MTLColorWriteMaskRed | MTLColorWriteMaskGreen},
		{types.ColorWriteMaskBlue | types.ColorWriteMaskAlpha, MTLColorWriteMaskBlue | MTLColorWriteMaskAlpha},
		{types.ColorWriteMaskAll, MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue | MTLColorWriteMaskAlpha},
	}

	for _, c := range cases {
		if got := colorWriteMaskToMTL(c.mask); got != c.want {
			t.Fatalf("colorWriteMaskToMTL(%v)=%v, want %v", c.mask, got, c.want)
		}
	}
}
