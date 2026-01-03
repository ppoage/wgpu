// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin

package metal

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unsafe"

	"github.com/gogpu/naga"
	"github.com/gogpu/naga/ir"
	"github.com/gogpu/naga/msl"
	"github.com/gogpu/wgpu/hal"
	"github.com/gogpu/wgpu/types"
)

// Device implements hal.Device for Metal.
type Device struct {
	raw          ID // id<MTLDevice>
	commandQueue ID // id<MTLCommandQueue>
	adapter      *Adapter
}

// newDevice creates a new Device from a Metal device.
func newDevice(adapter *Adapter) (*Device, error) {
	if adapter.raw == 0 {
		return nil, fmt.Errorf("metal: adapter has no device")
	}

	queue := MsgSend(adapter.raw, Sel("newCommandQueue"))
	if queue == 0 {
		return nil, fmt.Errorf("metal: failed to create command queue")
	}

	return &Device{
		raw:          adapter.raw,
		commandQueue: queue,
		adapter:      adapter,
	}, nil
}

// CreateBuffer creates a GPU buffer.
func (d *Device) CreateBuffer(desc *hal.BufferDescriptor) (hal.Buffer, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: buffer descriptor is nil")
	}
	if desc.Size == 0 {
		return nil, fmt.Errorf("metal: buffer size must be > 0")
	}

	var options MTLResourceOptions
	mapRead := desc.Usage&types.BufferUsageMapRead != 0
	mapWrite := desc.Usage&types.BufferUsageMapWrite != 0

	if mapRead || mapWrite {
		options = MTLResourceStorageModeShared
	} else {
		options = MTLResourceStorageModePrivate
	}

	if mapWrite && !mapRead {
		options |= MTLResourceCPUCacheModeWriteCombined
	}

	pool := NewAutoreleasePool()
	defer pool.Drain()

	raw := MsgSend(d.raw, Sel("newBufferWithLength:options:"),
		uintptr(desc.Size), uintptr(options))
	if raw == 0 {
		return nil, fmt.Errorf("metal: failed to create buffer")
	}

	if desc.Label != "" {
		label := NSString(desc.Label)
		_ = MsgSend(raw, Sel("setLabel:"), uintptr(label))
	}

	return &Buffer{
		raw:     raw,
		size:    desc.Size,
		usage:   desc.Usage,
		options: options,
		device:  d,
	}, nil
}

// DestroyBuffer destroys a GPU buffer.
func (d *Device) DestroyBuffer(buffer hal.Buffer) {
	mtlBuffer, ok := buffer.(*Buffer)
	if !ok || mtlBuffer == nil {
		return
	}
	if mtlBuffer.raw != 0 {
		Release(mtlBuffer.raw)
		mtlBuffer.raw = 0
	}
	mtlBuffer.device = nil
}

// CreateTexture creates a GPU texture.
func (d *Device) CreateTexture(desc *hal.TextureDescriptor) (hal.Texture, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: texture descriptor is nil")
	}
	if desc.Size.Width == 0 || desc.Size.Height == 0 {
		return nil, fmt.Errorf("metal: texture size must be > 0")
	}

	pool := NewAutoreleasePool()
	defer pool.Drain()

	texDesc := MsgSend(ID(GetClass("MTLTextureDescriptor")), Sel("new"))
	if texDesc == 0 {
		return nil, fmt.Errorf("metal: failed to create texture descriptor")
	}
	defer Release(texDesc)

	texType := textureTypeFromDimension(desc.Dimension, desc.SampleCount, desc.Size.DepthOrArrayLayers)
	_ = MsgSend(texDesc, Sel("setTextureType:"), uintptr(texType))

	pixelFormat := textureFormatToMTL(desc.Format)
	_ = MsgSend(texDesc, Sel("setPixelFormat:"), uintptr(pixelFormat))

	_ = MsgSend(texDesc, Sel("setWidth:"), uintptr(desc.Size.Width))
	_ = MsgSend(texDesc, Sel("setHeight:"), uintptr(desc.Size.Height))

	depth := desc.Size.DepthOrArrayLayers
	if depth == 0 {
		depth = 1
	}
	_ = MsgSend(texDesc, Sel("setDepth:"), uintptr(depth))

	mipLevels := desc.MipLevelCount
	if mipLevels == 0 {
		mipLevels = 1
	}
	_ = MsgSend(texDesc, Sel("setMipmapLevelCount:"), uintptr(mipLevels))

	sampleCount := desc.SampleCount
	if sampleCount == 0 {
		sampleCount = 1
	}
	_ = MsgSend(texDesc, Sel("setSampleCount:"), uintptr(sampleCount))

	usage := textureUsageToMTL(desc.Usage)
	_ = MsgSend(texDesc, Sel("setUsage:"), uintptr(usage))
	_ = MsgSend(texDesc, Sel("setStorageMode:"), uintptr(MTLStorageModePrivate))

	raw := MsgSend(d.raw, Sel("newTextureWithDescriptor:"), uintptr(texDesc))
	if raw == 0 {
		return nil, fmt.Errorf("metal: failed to create texture")
	}

	if desc.Label != "" {
		label := NSString(desc.Label)
		_ = MsgSend(raw, Sel("setLabel:"), uintptr(label))
	}

	return &Texture{
		raw:        raw,
		format:     desc.Format,
		width:      desc.Size.Width,
		height:     desc.Size.Height,
		depth:      depth,
		mipLevels:  mipLevels,
		samples:    sampleCount,
		dimension:  desc.Dimension,
		usage:      desc.Usage,
		device:     d,
		isExternal: false,
	}, nil
}

// DestroyTexture destroys a GPU texture.
func (d *Device) DestroyTexture(texture hal.Texture) {
	mtlTexture, ok := texture.(*Texture)
	if !ok || mtlTexture == nil {
		return
	}
	if mtlTexture.raw != 0 && !mtlTexture.isExternal {
		Release(mtlTexture.raw)
		mtlTexture.raw = 0
	}
	mtlTexture.device = nil
}

// CreateTextureView creates a view into a texture.
func (d *Device) CreateTextureView(texture hal.Texture, desc *hal.TextureViewDescriptor) (hal.TextureView, error) {
	var mtlTexture *Texture
	switch t := texture.(type) {
	case *Texture:
		mtlTexture = t
	case *SurfaceTexture:
		if t != nil {
			mtlTexture = t.texture
		}
	}
	if mtlTexture == nil {
		return nil, fmt.Errorf("metal: invalid texture")
	}
	if desc == nil {
		desc = &hal.TextureViewDescriptor{}
	}

	pool := NewAutoreleasePool()
	defer pool.Drain()

	format := desc.Format
	if format == types.TextureFormatUndefined {
		format = mtlTexture.format
	}
	pixelFormat := textureFormatToMTL(format)

	baseMip := desc.BaseMipLevel
	mipCount := desc.MipLevelCount
	if mipCount == 0 {
		// 0 means "all remaining mip levels" in WebGPU spec
		mipCount = mtlTexture.mipLevels - baseMip
	}

	baseLayer := desc.BaseArrayLayer
	layerCount := desc.ArrayLayerCount
	if layerCount == 0 {
		// 0 means "all remaining array layers" in WebGPU spec
		layerCount = mtlTexture.depth - baseLayer
		if layerCount == 0 {
			layerCount = 1
		}
	}

	var viewType MTLTextureType
	if desc.Dimension == types.TextureViewDimensionUndefined {
		viewType = textureTypeFromDimension(mtlTexture.dimension, mtlTexture.samples, mtlTexture.depth)
	} else {
		viewType = textureViewDimensionToMTL(desc.Dimension)
	}

	// Metal's newTextureViewWithPixelFormat:textureType:levels:slices: expects NSRange structs
	levelRange := NSRange{
		Location: NSUInteger(baseMip),
		Length:   NSUInteger(mipCount),
	}
	sliceRange := NSRange{
		Location: NSUInteger(baseLayer),
		Length:   NSUInteger(layerCount),
	}

	raw := msgSendID(mtlTexture.raw, Sel("newTextureViewWithPixelFormat:textureType:levels:slices:"),
		argUint64(uint64(pixelFormat)),
		argUint64(uint64(viewType)),
		argStruct(levelRange, nsRangeType),
		argStruct(sliceRange, nsRangeType),
	)
	if raw == 0 {
		return nil, fmt.Errorf("metal: failed to create texture view")
	}

	return &TextureView{raw: raw, texture: mtlTexture, device: d}, nil
}

// DestroyTextureView destroys a texture view.
func (d *Device) DestroyTextureView(view hal.TextureView) {
	mtlView, ok := view.(*TextureView)
	if !ok || mtlView == nil {
		return
	}
	if mtlView.raw != 0 {
		Release(mtlView.raw)
		mtlView.raw = 0
	}
	mtlView.device = nil
}

// CreateSampler creates a texture sampler.
func (d *Device) CreateSampler(desc *hal.SamplerDescriptor) (hal.Sampler, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: sampler descriptor is nil")
	}

	pool := NewAutoreleasePool()
	defer pool.Drain()

	sampDesc := MsgSend(ID(GetClass("MTLSamplerDescriptor")), Sel("new"))
	if sampDesc == 0 {
		return nil, fmt.Errorf("metal: failed to create sampler descriptor")
	}
	defer Release(sampDesc)

	_ = MsgSend(sampDesc, Sel("setMinFilter:"), uintptr(filterModeToMTL(desc.MinFilter)))
	_ = MsgSend(sampDesc, Sel("setMagFilter:"), uintptr(filterModeToMTL(desc.MagFilter)))
	_ = MsgSend(sampDesc, Sel("setMipFilter:"), uintptr(mipmapFilterModeToMTL(desc.MipmapFilter)))
	_ = MsgSend(sampDesc, Sel("setSAddressMode:"), uintptr(addressModeToMTL(desc.AddressModeU)))
	_ = MsgSend(sampDesc, Sel("setTAddressMode:"), uintptr(addressModeToMTL(desc.AddressModeV)))
	_ = MsgSend(sampDesc, Sel("setRAddressMode:"), uintptr(addressModeToMTL(desc.AddressModeW)))

	if desc.Anisotropy > 1 {
		_ = MsgSend(sampDesc, Sel("setMaxAnisotropy:"), uintptr(desc.Anisotropy))
	}

	if desc.Compare != types.CompareFunctionUndefined {
		_ = MsgSend(sampDesc, Sel("setCompareFunction:"), uintptr(compareFunctionToMTL(desc.Compare)))
	}

	raw := MsgSend(d.raw, Sel("newSamplerStateWithDescriptor:"), uintptr(sampDesc))
	if raw == 0 {
		return nil, fmt.Errorf("metal: failed to create sampler state")
	}

	return &Sampler{raw: raw, device: d}, nil
}

// DestroySampler destroys a sampler.
func (d *Device) DestroySampler(sampler hal.Sampler) {
	mtlSampler, ok := sampler.(*Sampler)
	if !ok || mtlSampler == nil {
		return
	}
	if mtlSampler.raw != 0 {
		Release(mtlSampler.raw)
		mtlSampler.raw = 0
	}
	mtlSampler.device = nil
}

// CreateBindGroupLayout creates a bind group layout.
func (d *Device) CreateBindGroupLayout(desc *hal.BindGroupLayoutDescriptor) (hal.BindGroupLayout, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: bind group layout descriptor is nil")
	}

	entries := append([]types.BindGroupLayoutEntry(nil), desc.Entries...)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Binding < entries[j].Binding
	})

	info := make(map[uint32]bindingInfo, len(entries))
	dynamicBindings := make([]uint32, 0)
	dynamicIndex := make(map[uint32]int)
	var bufferCount, textureCount, samplerCount uint32

	for _, entry := range entries {
		if _, exists := info[entry.Binding]; exists {
			return nil, fmt.Errorf("metal: duplicate bind group layout binding %d", entry.Binding)
		}

		kind, storage, err := classifyLayoutEntry(entry)
		if err != nil {
			return nil, err
		}

		var idx uint32
		switch kind {
		case bindingKindBuffer:
			idx = bufferCount
			bufferCount++
		case bindingKindTexture:
			idx = textureCount
			textureCount++
		case bindingKindSampler:
			idx = samplerCount
			samplerCount++
		default:
			return nil, fmt.Errorf("metal: unsupported binding kind for binding %d", entry.Binding)
		}

		hasDynamicOffset := entry.Buffer != nil && entry.Buffer.HasDynamicOffset
		if hasDynamicOffset {
			dynamicIndex[entry.Binding] = len(dynamicBindings)
			dynamicBindings = append(dynamicBindings, entry.Binding)
		}

		info[entry.Binding] = bindingInfo{
			kind:             kind,
			visibility:       entry.Visibility,
			index:            idx,
			hasDynamicOffset: hasDynamicOffset,
			storageTexture:   storage,
		}
	}

	return &BindGroupLayout{
		entries:         entries,
		bindingInfo:     info,
		dynamicBindings: dynamicBindings,
		dynamicIndex:    dynamicIndex,
		bufferCount:     bufferCount,
		textureCount:    textureCount,
		samplerCount:    samplerCount,
		device:          d,
	}, nil
}

// DestroyBindGroupLayout destroys a bind group layout.
func (d *Device) DestroyBindGroupLayout(layout hal.BindGroupLayout) {
	mtlLayout, ok := layout.(*BindGroupLayout)
	if !ok || mtlLayout == nil {
		return
	}
	mtlLayout.device = nil
}

// CreateBindGroup creates a bind group.
func (d *Device) CreateBindGroup(desc *hal.BindGroupDescriptor) (hal.BindGroup, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: bind group descriptor is nil")
	}

	layout, ok := desc.Layout.(*BindGroupLayout)
	if !ok || layout == nil {
		return nil, fmt.Errorf("metal: invalid bind group layout")
	}

	entries := append([]types.BindGroupEntry(nil), desc.Entries...)
	seen := make(map[uint32]struct{}, len(entries))
	for _, entry := range entries {
		if _, exists := seen[entry.Binding]; exists {
			return nil, fmt.Errorf("metal: duplicate bind group entry binding %d", entry.Binding)
		}
		seen[entry.Binding] = struct{}{}

		info, ok := layout.bindingInfo[entry.Binding]
		if !ok {
			return nil, fmt.Errorf("metal: bind group entry binding %d not in layout", entry.Binding)
		}

		kind, err := bindingKindFromResource(entry.Resource)
		if err != nil {
			return nil, err
		}
		if kind != info.kind {
			return nil, fmt.Errorf("metal: bind group entry binding %d has mismatched resource kind", entry.Binding)
		}
	}

	return &BindGroup{layout: layout, entries: entries, device: d}, nil
}

// DestroyBindGroup destroys a bind group.
func (d *Device) DestroyBindGroup(group hal.BindGroup) {
	mtlGroup, ok := group.(*BindGroup)
	if !ok || mtlGroup == nil {
		return
	}
	mtlGroup.device = nil
}

// CreatePipelineLayout creates a pipeline layout.
func (d *Device) CreatePipelineLayout(desc *hal.PipelineLayoutDescriptor) (hal.PipelineLayout, error) {
	if desc == nil {
		return nil, fmt.Errorf("metal: pipeline layout descriptor is nil")
	}

	layouts := make([]*BindGroupLayout, len(desc.BindGroupLayouts))
	for i, layout := range desc.BindGroupLayouts {
		mtlLayout, ok := layout.(*BindGroupLayout)
		if !ok || mtlLayout == nil {
			return nil, fmt.Errorf("metal: invalid bind group layout at index %d", i)
		}
		layouts[i] = mtlLayout
	}

	offsets := make([]groupOffsets, len(layouts))
	var bufferBase, textureBase, samplerBase uint32
	for i, layout := range layouts {
		offsets[i] = groupOffsets{
			buffer:  bufferBase,
			texture: textureBase,
			sampler: samplerBase,
		}
		bufferBase += layout.bufferCount
		textureBase += layout.textureCount
		samplerBase += layout.samplerCount
	}

	return &PipelineLayout{layouts: layouts, offsets: offsets, device: d}, nil
}

// DestroyPipelineLayout destroys a pipeline layout.
func (d *Device) DestroyPipelineLayout(layout hal.PipelineLayout) {
	mtlLayout, ok := layout.(*PipelineLayout)
	if !ok || mtlLayout == nil {
		return
	}
	mtlLayout.device = nil
}

// CreateShaderModule creates a shader module.
func (d *Device) CreateShaderModule(desc *hal.ShaderModuleDescriptor) (hal.ShaderModule, error) {
	// If WGSL source is provided, compile to MSL
	if desc.Source.WGSL != "" {
		// Parse WGSL to AST
		ast, err := naga.Parse(desc.Source.WGSL)
		if err != nil {
			return nil, fmt.Errorf("metal: failed to parse WGSL: %w", err)
		}

		// Lower AST to IR
		irModule, err := naga.LowerWithSource(ast, desc.Source.WGSL)
		if err != nil {
			return nil, fmt.Errorf("metal: failed to lower WGSL to IR: %w", err)
		}

		// Extract workgroup sizes from entry points for compute shaders
		workgroupSizes := extractWorkgroupSizes(irModule)

		// Compile IR to MSL
		mslSource, _, err := msl.Compile(irModule, msl.DefaultOptions())
		if err != nil {
			return nil, fmt.Errorf("metal: failed to compile to MSL: %w", err)
		}

		// Create NSString from MSL source
		mslString := NSString(mslSource)

		// Create MTLLibrary from source
		// MTLLibrary* newLibraryWithSource:options:error:
		var errorPtr ID
		library := MsgSend(d.raw, Sel("newLibraryWithSource:options:error:"),
			uintptr(mslString), 0, uintptr(unsafe.Pointer(&errorPtr)))

		if library == 0 {
			errMsg := "unknown error"
			if errorPtr != 0 {
				if details := formatNSError(errorPtr); details != "" {
					errMsg = details
				}
				// Object is autoreleased
			}
			return nil, fmt.Errorf("metal: failed to compile MSL: %s\nMSL:\n%s", errMsg, mslSource)
		}

		return &ShaderModule{
			source:         desc.Source,
			library:        library,
			device:         d,
			workgroupSizes: workgroupSizes,
		}, nil
	}

	// No WGSL source - just store the descriptor for later
	return &ShaderModule{source: desc.Source, device: d}, nil
}

func formatNSError(errObj ID) string {
	if errObj == 0 {
		return ""
	}
	parts := make([]string, 0, 4)
	if desc := GoString(MsgSend(errObj, Sel("localizedDescription"))); desc != "" {
		parts = append(parts, desc)
	}
	if reason := GoString(MsgSend(errObj, Sel("localizedFailureReason"))); reason != "" {
		parts = append(parts, reason)
	}
	if debug := GoString(MsgSend(errObj, Sel("debugDescription"))); debug != "" {
		parts = append(parts, debug)
	}
	if info := MsgSend(errObj, Sel("userInfo")); info != 0 {
		if infoDesc := GoString(MsgSend(info, Sel("description"))); infoDesc != "" {
			parts = append(parts, infoDesc)
		}
	}
	return strings.Join(parts, " | ")
}

// DestroyShaderModule destroys a shader module.
func (d *Device) DestroyShaderModule(module hal.ShaderModule) {
	mtlModule, ok := module.(*ShaderModule)
	if !ok || mtlModule == nil {
		return
	}
	if mtlModule.library != 0 {
		Release(mtlModule.library)
		mtlModule.library = 0
	}
	mtlModule.device = nil
}

type compiledLibrary struct {
	library        ID
	workgroupSizes map[string][3]uint32
	mslSource      string
}

func (d *Device) compileLibraryForPipeline(module *ShaderModule, layout *PipelineLayout) (*compiledLibrary, error) {
	if module == nil {
		return nil, fmt.Errorf("metal: shader module is nil")
	}
	if module.source.WGSL == "" {
		return nil, fmt.Errorf("metal: shader module missing WGSL source")
	}

	ast, err := naga.Parse(module.source.WGSL)
	if err != nil {
		return nil, fmt.Errorf("metal: failed to parse WGSL: %w", err)
	}

	irModule, err := naga.LowerWithSource(ast, module.source.WGSL)
	if err != nil {
		return nil, fmt.Errorf("metal: failed to lower WGSL to IR: %w", err)
	}

	if err := remapModuleBindings(irModule, layout); err != nil {
		return nil, err
	}

	workgroupSizes := extractWorkgroupSizes(irModule)

	mslSource, _, err := msl.Compile(irModule, msl.DefaultOptions())
	if err != nil {
		return nil, fmt.Errorf("metal: failed to compile to MSL: %w", err)
	}

	mslString := NSString(mslSource)

	var errorPtr ID
	library := MsgSend(d.raw, Sel("newLibraryWithSource:options:error:"),
		uintptr(mslString), 0, uintptr(unsafe.Pointer(&errorPtr)))
	if library == 0 {
		errMsg := "unknown error"
		if errorPtr != 0 {
			if details := formatNSError(errorPtr); details != "" {
				errMsg = details
			}
		}
		return nil, fmt.Errorf("metal: failed to compile MSL: %s\nMSL:\n%s", errMsg, mslSource)
	}

	return &compiledLibrary{
		library:        library,
		workgroupSizes: workgroupSizes,
		mslSource:      mslSource,
	}, nil
}

func remapModuleBindings(module *ir.Module, layout *PipelineLayout) error {
	if module == nil {
		return fmt.Errorf("metal: IR module is nil")
	}
	if layout == nil {
		for _, gv := range module.GlobalVariables {
			if gv.Binding != nil {
				return fmt.Errorf("metal: pipeline layout required for bound resources")
			}
		}
		return nil
	}

	for i := range module.GlobalVariables {
		gv := &module.GlobalVariables[i]
		if gv.Binding == nil {
			continue
		}

		if int(gv.Binding.Group) >= len(layout.layouts) {
			return fmt.Errorf("metal: bind group index %d out of range", gv.Binding.Group)
		}
		groupLayout := layout.layouts[gv.Binding.Group]
		if groupLayout == nil {
			return fmt.Errorf("metal: bind group layout %d is nil", gv.Binding.Group)
		}

		kind, storage, err := bindingKindFromGlobal(module, gv)
		if err != nil {
			return err
		}

		info, ok := groupLayout.bindingInfo[gv.Binding.Binding]
		if !ok {
			return fmt.Errorf("metal: binding %d not found in bind group %d", gv.Binding.Binding, gv.Binding.Group)
		}
		if info.kind != kind {
			return fmt.Errorf("metal: binding %d kind mismatch", gv.Binding.Binding)
		}
		if kind == bindingKindTexture {
			if storage && !info.storageTexture {
				return fmt.Errorf("metal: binding %d expects storage texture", gv.Binding.Binding)
			}
			if !storage && info.storageTexture {
				return fmt.Errorf("metal: binding %d expects sampled texture", gv.Binding.Binding)
			}
		}

		offsets := layout.offsets[gv.Binding.Group]
		switch kind {
		case bindingKindBuffer:
			gv.Binding.Binding = offsets.buffer + info.index
		case bindingKindTexture:
			gv.Binding.Binding = offsets.texture + info.index
		case bindingKindSampler:
			gv.Binding.Binding = offsets.sampler + info.index
		default:
			return fmt.Errorf("metal: unsupported binding kind for binding %d", gv.Binding.Binding)
		}
	}

	return nil
}

func bindingKindFromGlobal(module *ir.Module, gv *ir.GlobalVariable) (bindingKind, bool, error) {
	if gv == nil {
		return 0, false, fmt.Errorf("metal: global variable is nil")
	}
	if int(gv.Type) >= len(module.Types) {
		return 0, false, fmt.Errorf("metal: global variable type handle out of range")
	}

	switch inner := module.Types[gv.Type].Inner.(type) {
	case ir.SamplerType:
		return bindingKindSampler, false, nil
	case ir.ImageType:
		return bindingKindTexture, inner.Class == ir.ImageClassStorage, nil
	default:
		if gv.Space == ir.SpaceUniform || gv.Space == ir.SpaceStorage {
			return bindingKindBuffer, false, nil
		}
	}

	return 0, false, fmt.Errorf("metal: unsupported binding type for global %s", gv.Name)
}

// CreateRenderPipeline creates a render pipeline.
func (d *Device) CreateRenderPipeline(desc *hal.RenderPipelineDescriptor) (hal.RenderPipeline, error) {
	pool := NewAutoreleasePool()
	defer pool.Drain()

	layout, ok := desc.Layout.(*PipelineLayout)
	if !ok || layout == nil {
		return nil, fmt.Errorf("metal: invalid pipeline layout")
	}

	// Get shader modules
	vertexModule, ok := desc.Vertex.Module.(*ShaderModule)
	if !ok || vertexModule == nil {
		return nil, fmt.Errorf("metal: invalid vertex shader module")
	}

	var fragmentModule *ShaderModule
	if desc.Fragment != nil {
		fragmentModule, ok = desc.Fragment.Module.(*ShaderModule)
		if !ok || fragmentModule == nil {
			return nil, fmt.Errorf("metal: invalid fragment shader module")
		}
	}

	vertexCompiled, err := d.compileLibraryForPipeline(vertexModule, layout)
	if err != nil {
		return nil, err
	}
	defer Release(vertexCompiled.library)

	fragmentCompiled := vertexCompiled
	if fragmentModule != nil && fragmentModule != vertexModule {
		fragmentCompiled, err = d.compileLibraryForPipeline(fragmentModule, layout)
		if err != nil {
			return nil, err
		}
		defer Release(fragmentCompiled.library)
	}

	// Create pipeline descriptor
	pipelineDesc := MsgSend(ID(GetClass("MTLRenderPipelineDescriptor")), Sel("new"))
	if pipelineDesc == 0 {
		return nil, fmt.Errorf("metal: failed to create pipeline descriptor")
	}
	defer Release(pipelineDesc)

	// Set label if provided
	if desc.Label != "" {
		label := NSString(desc.Label)
		_ = MsgSend(pipelineDesc, Sel("setLabel:"), uintptr(label))
	}

	// Get vertex function from library
	vertexFuncName := NSString(desc.Vertex.EntryPoint)
	vertexFunc := MsgSend(vertexCompiled.library, Sel("newFunctionWithName:"), uintptr(vertexFuncName))
	if vertexFunc == 0 {
		return nil, fmt.Errorf("metal: vertex function '%s' not found", desc.Vertex.EntryPoint)
	}
	defer Release(vertexFunc)

	// Set vertex function
	_ = MsgSend(pipelineDesc, Sel("setVertexFunction:"), uintptr(vertexFunc))

	if len(desc.Vertex.Buffers) > 0 {
		vertexDesc := MsgSend(ID(GetClass("MTLVertexDescriptor")), Sel("vertexDescriptor"))
		if vertexDesc == 0 {
			return nil, fmt.Errorf("metal: failed to create vertex descriptor")
		}

		attributes := MsgSend(vertexDesc, Sel("attributes"))
		layouts := MsgSend(vertexDesc, Sel("layouts"))
		if attributes == 0 || layouts == 0 {
			return nil, fmt.Errorf("metal: failed to access vertex descriptor arrays")
		}

		for i, buf := range desc.Vertex.Buffers {
			layout := MsgSend(layouts, Sel("objectAtIndexedSubscript:"), uintptr(i))
			if layout == 0 {
				return nil, fmt.Errorf("metal: vertex layout %d unavailable", i)
			}

			stepFunc, ok := vertexStepModeToMTL(buf.StepMode)
			if !ok {
				return nil, fmt.Errorf("metal: unsupported vertex step mode %d", buf.StepMode)
			}

			_ = MsgSend(layout, Sel("setStride:"), uintptr(buf.ArrayStride))
			_ = MsgSend(layout, Sel("setStepFunction:"), uintptr(stepFunc))
			_ = MsgSend(layout, Sel("setStepRate:"), uintptr(1))

			for _, attr := range buf.Attributes {
				attrDesc := MsgSend(attributes, Sel("objectAtIndexedSubscript:"), uintptr(attr.ShaderLocation))
				if attrDesc == 0 {
					return nil, fmt.Errorf("metal: vertex attribute %d unavailable", attr.ShaderLocation)
				}

				format, ok := vertexFormatToMTL(attr.Format)
				if !ok {
					return nil, fmt.Errorf("metal: unsupported vertex format %d", attr.Format)
				}

				_ = MsgSend(attrDesc, Sel("setFormat:"), uintptr(format))
				_ = MsgSend(attrDesc, Sel("setOffset:"), uintptr(attr.Offset))
				_ = MsgSend(attrDesc, Sel("setBufferIndex:"), uintptr(i))
			}
		}

		_ = MsgSend(pipelineDesc, Sel("setVertexDescriptor:"), uintptr(vertexDesc))
	}

	// Get and set fragment function if present
	if fragmentModule != nil && desc.Fragment != nil {
		fragmentFuncName := NSString(desc.Fragment.EntryPoint)
		fragmentFunc := MsgSend(fragmentCompiled.library, Sel("newFunctionWithName:"), uintptr(fragmentFuncName))
		if fragmentFunc == 0 {
			return nil, fmt.Errorf("metal: fragment function '%s' not found", desc.Fragment.EntryPoint)
		}
		defer Release(fragmentFunc)

		_ = MsgSend(pipelineDesc, Sel("setFragmentFunction:"), uintptr(fragmentFunc))

		// Configure color attachments
		colorAttachments := MsgSend(pipelineDesc, Sel("colorAttachments"))
		for i, target := range desc.Fragment.Targets {
			attachment := MsgSend(colorAttachments, Sel("objectAtIndexedSubscript:"), uintptr(i))
			if attachment == 0 {
				continue
			}

			// Set pixel format
			pixelFormat := textureFormatToMTL(target.Format)
			_ = MsgSend(attachment, Sel("setPixelFormat:"), uintptr(pixelFormat))

			// Set write mask
			_ = MsgSend(attachment, Sel("setWriteMask:"), uintptr(colorWriteMaskToMTL(target.WriteMask)))

			// Configure blending if present
			if target.Blend != nil {
				_ = MsgSend(attachment, Sel("setBlendingEnabled:"), uintptr(1))
				_ = MsgSend(attachment, Sel("setSourceRGBBlendFactor:"), uintptr(blendFactorToMTL(target.Blend.Color.SrcFactor)))
				_ = MsgSend(attachment, Sel("setDestinationRGBBlendFactor:"), uintptr(blendFactorToMTL(target.Blend.Color.DstFactor)))
				_ = MsgSend(attachment, Sel("setRgbBlendOperation:"), uintptr(blendOperationToMTL(target.Blend.Color.Operation)))
				_ = MsgSend(attachment, Sel("setSourceAlphaBlendFactor:"), uintptr(blendFactorToMTL(target.Blend.Alpha.SrcFactor)))
				_ = MsgSend(attachment, Sel("setDestinationAlphaBlendFactor:"), uintptr(blendFactorToMTL(target.Blend.Alpha.DstFactor)))
				_ = MsgSend(attachment, Sel("setAlphaBlendOperation:"), uintptr(blendOperationToMTL(target.Blend.Alpha.Operation)))
			}
		}
	}

	// Set sample count
	sampleCount := desc.Multisample.Count
	if sampleCount == 0 {
		sampleCount = 1
	}
	_ = MsgSend(pipelineDesc, Sel("setSampleCount:"), uintptr(sampleCount))

	// Create pipeline state
	var errorPtr ID
	pipelineState := MsgSend(d.raw, Sel("newRenderPipelineStateWithDescriptor:error:"),
		uintptr(pipelineDesc), uintptr(unsafe.Pointer(&errorPtr)))

	if pipelineState == 0 {
		errMsg := "unknown error"
		if errorPtr != 0 {
			errDesc := MsgSend(errorPtr, Sel("localizedDescription"))
			if errDesc != 0 {
				errMsg = GoString(errDesc)
			}
			// Object is autoreleased
		}
		return nil, fmt.Errorf("metal: failed to create pipeline state: %s", errMsg)
	}

	return &RenderPipeline{raw: pipelineState, layout: layout, device: d}, nil
}

// DestroyRenderPipeline destroys a render pipeline.
func (d *Device) DestroyRenderPipeline(pipeline hal.RenderPipeline) {
	mtlPipeline, ok := pipeline.(*RenderPipeline)
	if !ok || mtlPipeline == nil {
		return
	}
	if mtlPipeline.raw != 0 {
		Release(mtlPipeline.raw)
		mtlPipeline.raw = 0
	}
	mtlPipeline.layout = nil
	mtlPipeline.device = nil
}

// CreateComputePipeline creates a compute pipeline.
func (d *Device) CreateComputePipeline(desc *hal.ComputePipelineDescriptor) (hal.ComputePipeline, error) {
	pool := NewAutoreleasePool()
	defer pool.Drain()

	layout, ok := desc.Layout.(*PipelineLayout)
	if !ok || layout == nil {
		return nil, fmt.Errorf("metal: invalid pipeline layout")
	}

	// Get shader module
	computeModule, ok := desc.Compute.Module.(*ShaderModule)
	if !ok || computeModule == nil {
		return nil, fmt.Errorf("metal: invalid compute shader module")
	}

	compiled, err := d.compileLibraryForPipeline(computeModule, layout)
	if err != nil {
		return nil, err
	}
	defer Release(compiled.library)

	// Get compute function from library
	funcName := NSString(desc.Compute.EntryPoint)
	computeFunc := MsgSend(compiled.library, Sel("newFunctionWithName:"), uintptr(funcName))
	if computeFunc == 0 {
		return nil, fmt.Errorf("metal: compute function '%s' not found", desc.Compute.EntryPoint)
	}
	defer Release(computeFunc)

	// Create compute pipeline state
	var errorPtr ID
	pipelineState := MsgSend(d.raw, Sel("newComputePipelineStateWithFunction:error:"),
		uintptr(computeFunc), uintptr(unsafe.Pointer(&errorPtr)))

	if pipelineState == 0 {
		errMsg := "unknown error"
		if errorPtr != 0 {
			errDesc := MsgSend(errorPtr, Sel("localizedDescription"))
			if errDesc != 0 {
				errMsg = GoString(errDesc)
			}
			// Object is autoreleased
		}
		return nil, fmt.Errorf("metal: failed to create compute pipeline state: %s", errMsg)
	}

	// Get workgroup size from shader module metadata
	workgroupSize := workgroupSizeForEntry(compiled.workgroupSizes, desc.Compute.EntryPoint)

	return &ComputePipeline{
		raw:           pipelineState,
		layout:        layout,
		device:        d,
		workgroupSize: workgroupSize,
	}, nil
}

// getWorkgroupSize retrieves workgroup size for a compute entry point.
// Falls back to default {64, 1, 1} if not found.
func getWorkgroupSize(module *ShaderModule, entryPoint string) MTLSize {
	if module.workgroupSizes != nil {
		if size, ok := module.workgroupSizes[entryPoint]; ok {
			return MTLSize{
				Width:  NSUInteger(size[0]),
				Height: NSUInteger(size[1]),
				Depth:  NSUInteger(size[2]),
			}
		}
	}
	// Default fallback
	return MTLSize{Width: 64, Height: 1, Depth: 1}
}

func workgroupSizeForEntry(workgroupSizes map[string][3]uint32, entryPoint string) MTLSize {
	if workgroupSizes != nil {
		if size, ok := workgroupSizes[entryPoint]; ok {
			return MTLSize{
				Width:  NSUInteger(size[0]),
				Height: NSUInteger(size[1]),
				Depth:  NSUInteger(size[2]),
			}
		}
	}
	return MTLSize{Width: 64, Height: 1, Depth: 1}
}

// DestroyComputePipeline destroys a compute pipeline.
func (d *Device) DestroyComputePipeline(pipeline hal.ComputePipeline) {
	mtlPipeline, ok := pipeline.(*ComputePipeline)
	if !ok || mtlPipeline == nil {
		return
	}
	if mtlPipeline.raw != 0 {
		Release(mtlPipeline.raw)
		mtlPipeline.raw = 0
	}
	mtlPipeline.layout = nil
	mtlPipeline.device = nil
}

// CreateCommandEncoder creates a command encoder.
func (d *Device) CreateCommandEncoder(desc *hal.CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	pool := NewAutoreleasePool()
	cmdBuffer := MsgSend(d.commandQueue, Sel("commandBuffer"))
	if cmdBuffer == 0 {
		pool.Drain()
		return nil, fmt.Errorf("metal: failed to create command buffer")
	}
	Retain(cmdBuffer)
	label := ""
	if desc != nil {
		label = desc.Label
	}
	return &CommandEncoder{device: d, cmdBuffer: cmdBuffer, pool: pool, label: label}, nil
}

// CreateFence creates a synchronization fence.
func (d *Device) CreateFence() (hal.Fence, error) {
	event := MsgSend(d.raw, Sel("newEvent"))
	if event == 0 {
		return nil, fmt.Errorf("metal: failed to create event")
	}
	return &Fence{event: event, value: 0, device: d}, nil
}

// DestroyFence destroys a fence.
func (d *Device) DestroyFence(fence hal.Fence) {
	mtlFence, ok := fence.(*Fence)
	if !ok || mtlFence == nil {
		return
	}
	if mtlFence.event != 0 {
		Release(mtlFence.event)
		mtlFence.event = 0
	}
	mtlFence.device = nil
}

// Wait waits for a fence to reach the specified value.
func (d *Device) Wait(fence hal.Fence, value uint64, timeout time.Duration) (bool, error) {
	mtlFence, ok := fence.(*Fence)
	if !ok || mtlFence == nil {
		return false, fmt.Errorf("metal: invalid fence")
	}
	if mtlFence.value >= value {
		return true, nil
	}
	return false, nil
}

// Destroy releases the device.
func (d *Device) Destroy() {
	if d.commandQueue != 0 {
		Release(d.commandQueue)
		d.commandQueue = 0
	}
}

// extractWorkgroupSizes extracts workgroup sizes from IR module entry points.
// Returns a map from entry point name to workgroup size [x, y, z].
func extractWorkgroupSizes(module *ir.Module) map[string][3]uint32 {
	if module == nil {
		return nil
	}
	result := make(map[string][3]uint32)
	for _, ep := range module.EntryPoints {
		if ep.Stage == ir.StageCompute {
			result[ep.Name] = ep.Workgroup
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
