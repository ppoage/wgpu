// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"fmt"

	"github.com/gogpu/wgpu/types"
)

type bindingKind uint8

const (
	bindingKindBuffer bindingKind = iota
	bindingKindTexture
	bindingKindSampler
)

type bindingInfo struct {
	kind             bindingKind
	visibility       types.ShaderStages
	index            uint32
	hasDynamicOffset bool
	storageTexture   bool
}

type groupOffsets struct {
	buffer  uint32
	texture uint32
	sampler uint32
}

func classifyLayoutEntry(entry types.BindGroupLayoutEntry) (bindingKind, bool, error) {
	count := 0
	var kind bindingKind
	storage := false

	if entry.Buffer != nil {
		kind = bindingKindBuffer
		count++
	}
	if entry.Sampler != nil {
		kind = bindingKindSampler
		count++
	}
	if entry.Texture != nil {
		kind = bindingKindTexture
		count++
	}
	if entry.Storage != nil {
		kind = bindingKindTexture
		storage = true
		count++
	}

	if count != 1 {
		return 0, false, fmt.Errorf("metal: bind group layout entry %d must have exactly one binding type", entry.Binding)
	}

	return kind, storage, nil
}

func bindingKindFromResource(resource types.BindingResource) (bindingKind, error) {
	switch resource.(type) {
	case types.BufferBinding:
		return bindingKindBuffer, nil
	case types.SamplerBinding:
		return bindingKindSampler, nil
	case types.TextureViewBinding:
		return bindingKindTexture, nil
	default:
		return 0, fmt.Errorf("metal: unsupported binding resource type %T", resource)
	}
}
