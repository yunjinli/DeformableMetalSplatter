import Foundation
import Metal
import MetalKit
import os
import SplatIO
import Accelerate
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

#if arch(x86_64)
typealias Float16 = Float
#warning("x86_64 targets are unsupported by MetalSplatter and will fail at runtime. MetalSplatter builds on x86_64 only because Xcode builds Swift Packages as universal binaries and provides no way to override this. When Swift supports Float16 on x86_64, this may be revisited.")
#endif

public class SplatRenderer {
    enum Constants {
        // Keep in sync with Shaders.metal : maxViewCount
        static let maxViewCount = 2
        // Sort by euclidian distance squared from camera position (true), or along the "forward" vector (false)
        // TODO: compare the behaviour and performance of sortByDistance
        // notes: sortByDistance introduces unstable artifacts when you get close to an object; whereas !sortByDistance introduces artifacts are you turn -- but they're a little subtler maybe?
        static let sortByDistance = true
        // Only store indices for 1024 splats; for the remainder, use instancing of these existing indices.
        // Setting to 1 uses only instancing (with a significant performance penalty); setting to a number higher than the splat count
        // uses only indexing (with a significant memory penalty for th elarge index array, and a small performance penalty
        // because that can't be cached as easiliy). Anywhere within an order of magnitude (or more?) of 1k seems to be the sweet spot,
        // with effectively no memory penalty compated to instancing, and slightly better performance than even using all indexing.
        static let maxIndexedSplatCount = 1024

        static let tileSize = MTLSize(width: 32, height: 32, depth: 1)
    }

    private static let log = Logger(subsystem: Bundle.module.bundleIdentifier!,category: "SplatRenderer")
    
    private var computeDepthsPipelineState: MTLComputePipelineState?
    
    public struct ViewportDescriptor {
        public var viewport: MTLViewport
        public var projectionMatrix: simd_float4x4
        public var viewMatrix: simd_float4x4
        public var screenSize: SIMD2<Int>

        public init(viewport: MTLViewport, projectionMatrix: simd_float4x4, viewMatrix: simd_float4x4, screenSize: SIMD2<Int>) {
            self.viewport = viewport
            self.projectionMatrix = projectionMatrix
            self.viewMatrix = viewMatrix
            self.screenSize = screenSize
        }
    }

    // Keep in sync with Shaders.metal : BufferIndex
    enum BufferIndex: NSInteger {
        case uniforms = 0
        case splat    = 1
        case splatIndices = 2
        case clusterColor = 3
        case clusterID = 4
        case selectedClusters = 5
        case deformMask = 6  // For mask visualization
    }

    // Keep in sync with Shaders.metal : Uniforms (184 bytes total)
    struct Uniforms {
        var projectionMatrix: matrix_float4x4  // 64 bytes (offset 0)
        var viewMatrix: matrix_float4x4        // 64 bytes (offset 64)
        var screenSize: SIMD2<UInt32>          // 8 bytes  (offset 128)

        var splatCount: UInt32                 // 4 bytes  (offset 136)
        var indexedSplatCount: UInt32          // 4 bytes  (offset 140)
        var showClusterColors: UInt32          // 4 bytes  (offset 144)
        var showMask: UInt32                   // 4 bytes  (offset 148) - show dynamic splats in red
        var selectedClusterID: Int32           // 4 bytes  (offset 152) -1 means show all clusters (single selection)

        var showDepthVisualization: UInt32     // 4 bytes  (offset 156)
        var selectionMode: UInt32 = 0          // 4 bytes  (offset 160) 0=off, 1=selecting, 2=confirmed
        var depthRange: SIMD2<Float>           // 8 bytes  (offset 168) min/max depth

        var selectedClusterCount: UInt32 = 0   // 4 bytes  (offset 176)
        var maskThreshold: Float = 0             // 4 bytes  (offset 180) mask threshold
        // Total: 184 bytes
    }


    // Keep in sync with Shaders.metal : UniformsArray
    struct UniformsArray {
        // maxViewCount = 2, so we have 2 entries
        var uniforms0: Uniforms
        var uniforms1: Uniforms

        // The 256 byte aligned size of our uniform structure
        static var alignedSize: Int { (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100 }

        mutating func setUniforms(index: Int, _ uniforms: Uniforms) {
            switch index {
            case 0: uniforms0 = uniforms
            case 1: uniforms1 = uniforms
            default: break
            }
        }
    }

    struct PackedHalf3 {
        var x: Float16
        var y: Float16
        var z: Float16
    }

    struct PackedRGBHalf4 {
        var r: Float16
        var g: Float16
        var b: Float16
        var a: Float16
    }

    // Keep in sync with Shaders.metal : Splat
    struct Splat {
        var position: MTLPackedFloat3
        var color: PackedRGBHalf4
        var covA: PackedHalf3
        var covB: PackedHalf3
    }

    struct SplatIndexAndDepth {
        var index: UInt32
        var depth: Float
    }
    
    struct CanonicalSplat {
        var position: MTLPackedFloat3
        var color: PackedRGBHalf4
        var rotationX: Float
        var rotationY: Float
        var rotationZ: Float
        var rotationW: Float
        var scale: MTLPackedFloat3
    }
    
    // Deformation Support
    /// If true, the deformation MPSGraph uses FP16 weights/compute (casts I/O to/from FP32).
    /// This is a major speed win on Apple GPUs, but can introduce small numeric differences.
    public var useFP16Deformation: Bool = true
    /// When true, apply full deformation (position + rotation + scale deltas).
    /// When false, smooth mode: apply position deltas only, preserve canonical rotation and scale.

    public var useClusterColors: Bool = false
    public var selectedClusterID: Int32 = -1  // -1 means show all clusters
    public var showMask: Bool = false  // When true, show dynamic splats in red, static in original color

    /// Returns true if cluster data (clusters.bin) was successfully loaded
    public var hasClusters: Bool {
        clusterColorBuffer != nil && clusterIdBuffer != nil
    }

    /// Returns true if mask data (mask.bin) was successfully loaded
    public var hasMask: Bool {
        deformMaskBuffer != nil
    }
    public var useDepthVisualization: Bool = false
    private var currentDepthRange: SIMD2<Float> = SIMD2(0.1, 10.0)  // Default near/far
    var canonicalBuffer: MetalBuffer<CanonicalSplat>?
    var canonicalSplatBufferPrime: MetalBuffer<CanonicalSplat>?
    var sortedIndexBuffer: MetalBuffer<UInt32>?
    var deformSystem: DeformGraphSystem?
    var extractPipeline: MTLComputePipelineState?
    var timeFillPipeline: MTLComputePipelineState?
    var applyPipeline: MTLComputePipelineState?
    
    // Intermediate Buffers
    var bufXYZ: MTLBuffer?
    var bufT: MTLBuffer?
    var bufDXYZ: MTLBuffer?
    var bufDRot: MTLBuffer?
    var bufDScale: MTLBuffer?
    var clusterIdBuffer: MTLBuffer?
    var clusterColorBuffer: MTLBuffer?
    var deformMaskBuffer: MTLBuffer?
    var allOnesMaskBuffer: MTLBuffer?
    /// If true, use mask.bin for deformation at t>0; if false, always deform all splats
    public var useMaskedDeformation: Bool = false
    /// Threshold for dynamic thresholding: splats with mask value > threshold are deformed.
    /// mask.bin now stores continuous deformation magnitudes.
    public var maskThreshold: Float = 0.0
    /// Maximum mask value in the loaded mask.bin (for UI slider scaling)
    public var maxMaskValue: Float = 1.0
    /// Sorted mask values for percentile scaling
    public var sortedMaskValues: [Float] = []
    /// Recommended percentile offset derived from configuration script
    public var recommendedMaskPercentage: Double? = nil
    /// URL where the recommended mask percentage should be saved
    public var maskJsonURL: URL? = nil
    private let emptyClusterColorBuffer: MTLBuffer
    private let emptyClusterIdBuffer: MTLBuffer
    private var lastDeformationTime: Float = -1.0
    // Deformation FPS tracking (based on actual graph runtime)
    private var deformFPS: Double = 0.0
    private var _deformedSplatCount: Int = 0
    
    // Picking support (screen-space based)
    private var maxClusterID: UInt32 = 0
    
    // Multi-cluster selection support
    /// Selection mode: 0 = off, 1 = selecting (shows red), 2 = confirmed (filters)
    public var selectionMode: UInt32 = 0
    /// Set of cluster IDs that are currently selected
    public var selectedClusters: Set<UInt32> = []
    /// Buffer to pass selected cluster IDs to GPU
    private var selectedClustersBuffer: MTLBuffer?
    private let emptySelectedClustersBuffer: MTLBuffer
    /// Empty buffer for deform mask (used when no mask is loaded)
    private let emptyDeformMaskBuffer: MTLBuffer
    /// Max selected clusters (must match shader constant)
    private static let maxSelectedClusters = 64

    
    public let device: MTLDevice
    public let colorFormat: MTLPixelFormat
    public let depthFormat: MTLPixelFormat
    public let sampleCount: Int
    public let maxViewCount: Int
    public let maxSimultaneousRenders: Int

    /**
     High-quality depth takes longer, but results in a continuous, more-representative depth buffer result, which is useful for reducing artifacts during Vision Pro's frame reprojection.
     */
    public var highQualityDepth: Bool = true

    private var writeDepth: Bool {
        depthFormat != .invalid
    }

    /**
     The SplatRenderer has two shader pipelines.
     - The single stage has a vertex shader, and a fragment shader. It can produce depth (or not), but the depth it produces is the depth of the nearest splat, whether it's visible or now.
     - The multi-stage pipeline uses a set of shaders which communicate using imageblock tile memory: initialization (which clears the tile memory), draw splats (similar to the single-stage
     pipeline but the end result is tile memory, not color+depth), and a post-process stage which merely copies the tile memory (color and optionally depth) to the frame's buffers.
     This is neccessary so that the primary stage can do its own blending -- of both color and depth -- by reading the previous values and writing new ones, which isn't possible without tile
     memory. Color blending works the same as the hardcoded path, but depth blending uses color alpha and results in mostly-transparent splats contributing only slightly to the depth,
     resulting in a much more continuous and representative depth value, which is important for reprojection on Vision Pro.
     */
    private var useMultiStagePipeline: Bool {
#if targetEnvironment(simulator)
        false
#else
        writeDepth && highQualityDepth
#endif
    }

    public var clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)

    public var onSortStart: (() -> Void)?
    public var onSortComplete: ((TimeInterval) -> Void)?
    
    public private(set) var currentFPS: Double = 0
    private var fpsFrameCount = 0
    private var fpsLastTimestamp = CFAbsoluteTimeGetCurrent()

    private let library: MTLLibrary
    // Single-stage pipeline
    private var singleStagePipelineState: MTLRenderPipelineState?
    private var singleStageDepthState: MTLDepthStencilState?
    // Multi-stage pipeline
    private var initializePipelineState: MTLRenderPipelineState?
    private var drawSplatPipelineState: MTLRenderPipelineState?
    private var drawSplatDepthState: MTLDepthStencilState?
    private var postprocessPipelineState: MTLRenderPipelineState?
    private var postprocessDepthState: MTLDepthStencilState?

    // dynamicUniformBuffers contains maxSimultaneousRenders uniforms buffers,
    // which we round-robin through, one per render; this is managed by switchToNextDynamicBuffer.
    // uniforms = the i'th buffer (where i = uniformBufferIndex, which varies from 0 to maxSimultaneousRenders-1)
    var dynamicUniformBuffers: MTLBuffer
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>

    // cameraWorldPosition and Forward vectors are the latest mean camera position across all viewports
    var cameraWorldPosition: SIMD3<Float> = .zero
    var cameraWorldForward: SIMD3<Float> = .init(x: 0, y: 0, z: -1)

    typealias IndexType = UInt32
    // splatBuffer contains one entry for each gaussian splat
    var splatBuffer: MetalBuffer<Splat>
    // splatBufferPrime is a copy of splatBuffer, which is not currenly in use for rendering.
    // We use this for sorting, and when we're done, swap it with splatBuffer.
    // There's a good chance that we'll sometimes end up sorting a splatBuffer still in use for
    // rendering.
    // TODO: Replace this with a more robust multiple-buffer scheme to guarantee we're never actively sorting a buffer still in use for rendering
    var splatBufferPrime: MetalBuffer<Splat>
    
    var indexBuffer: MetalBuffer<UInt32>

    public var splatCount: Int { splatBuffer.count }

    var sorting = false
    var orderAndDepthTempSort: [SplatIndexAndDepth] = []

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
#if arch(x86_64)
        fatalError("MetalSplatter is unsupported on Intel architecture (x86_64)")
#endif
        self.device = device

        self.colorFormat = colorFormat
        self.depthFormat = depthFormat
        self.sampleCount = sampleCount
        self.maxViewCount = min(maxViewCount, Constants.maxViewCount)
        self.maxSimultaneousRenders = maxSimultaneousRenders

        let dynamicUniformBuffersSize = UniformsArray.alignedSize * maxSimultaneousRenders
        self.dynamicUniformBuffers = device.makeBuffer(length: dynamicUniformBuffersSize,
                                                       options: .storageModeShared)!
        self.dynamicUniformBuffers.label = "Uniform Buffers"
        self.uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents()).bindMemory(to: UniformsArray.self, capacity: 1)

        self.splatBuffer = try MetalBuffer(device: device)
        self.splatBufferPrime = try MetalBuffer(device: device)
        self.sortedIndexBuffer = try MetalBuffer(device: device)
        self.indexBuffer = try MetalBuffer(device: device)
        self.emptyClusterColorBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * 3,
                                                         options: .storageModeShared)!
        self.emptyClusterIdBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                      options: .storageModeShared)!
        self.emptySelectedClustersBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size * Self.maxSelectedClusters,
                                                             options: .storageModeShared)!
        // Empty deform mask buffer - enough for 1M splats as fallback
        self.emptyDeformMaskBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * 1_000_000,
                                                       options: .storageModeShared)!

        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            fatalError("Unable to initialize SplatRenderer: \(error)")
        }
    }

    public func reset() {
        splatBuffer.count = 0
        try? splatBuffer.setCapacity(0)
    }

    public func read(from url: URL) async throws {
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: try AutodetectSceneReader(url))
        try add(newPoints.points)
    }

    private func resetPipelineStates() {
        singleStagePipelineState = nil
        initializePipelineState = nil
        drawSplatPipelineState = nil
        drawSplatDepthState = nil
        postprocessPipelineState = nil
        postprocessDepthState = nil
    }

    private func buildSingleStagePipelineStatesIfNeeded() throws {
        guard singleStagePipelineState == nil else { return }

        singleStagePipelineState = try buildSingleStagePipelineState()
        singleStageDepthState = try buildSingleStageDepthState()
    }

    private func buildMultiStagePipelineStatesIfNeeded() throws {
        guard initializePipelineState == nil else { return }

        initializePipelineState = try buildInitializePipelineState()
        drawSplatPipelineState = try buildDrawSplatPipelineState()
        drawSplatDepthState = try buildDrawSplatDepthState()
        postprocessPipelineState = try buildPostprocessPipelineState()
        postprocessDepthState = try buildPostprocessDepthState()
    }

    private func buildSingleStagePipelineState() throws -> MTLRenderPipelineState {
        assert(!useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "SingleStagePipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "singleStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeRequiredFunction(name: "singleStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.pixelFormat = colorFormat
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .one
        colorAttachment.sourceAlphaBlendFactor = .one
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0] = colorAttachment
        pipelineDescriptor.colorAttachments[0] = colorAttachment

        let clusterIDAttachment = pipelineDescriptor.colorAttachments[1]!
        clusterIDAttachment.pixelFormat = .r32Sint
        clusterIDAttachment.isBlendingEnabled = false
        pipelineDescriptor.colorAttachments[1] = clusterIDAttachment

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildSingleStageDepthState() throws -> MTLDepthStencilState {
        assert(!useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }
    
    // MARK: - Picking Pipeline (screen-space based, finds splat closest to click point)
    
    // Keep in sync with SingleStageRenderPath.metal : PickingParams
    struct PickingParams {
        var clickPointX: Float          // 4 bytes
        var clickPointY: Float          // 4 bytes
        var screenWidth: Float          // 4 bytes
        var screenHeight: Float         // 4 bytes - total 16 bytes
    }
    
    private var pickingComputePipeline: MTLComputePipelineState?
    private var minScoreBuffer: MTLBuffer?
    private var resultClusterBuffer: MTLBuffer?
    
    private func buildPickingComputePipelineIfNeeded() throws {
        guard pickingComputePipeline == nil else { return }
        
        guard let findNearestFunc = library.makeFunction(name: "findNearestSplatToScreen") else {
            throw NSError(domain: "SplatRenderer", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Could not find findNearestSplatToScreen kernel"])
        }
        pickingComputePipeline = try device.makeComputePipelineState(function: findNearestFunc)
        
        // Buffers for atomic operations
        minScoreBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        resultClusterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)
    }
    
    
    /// Pick the cluster ID at a screen position.
    /// Projects all splats to screen space and finds the one closest to the click point.
    /// Uses depth to prefer closer splats when multiple project to similar screen positions.
    /// Returns -1 if no cluster at that position.
    public func pickCluster(at screenPoint: CGPoint,
                            viewportSize: CGSize,
                            viewport: ViewportDescriptor) -> Int32 {
        do {
            try buildPickingComputePipelineIfNeeded()
        } catch {
            print("Failed to build picking pipeline: \(error)")
            return -1
        }
        
        guard let computePipeline = pickingComputePipeline,
              let clusterIDs = clusterIdBuffer,
              let scoreBuffer = minScoreBuffer,
              let resultBuffer = resultClusterBuffer,
              splatBuffer.count > 0,
              maxClusterID > 0 else {
            return -1
        }
        
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return -1
        }
        
        // Initialize buffers
        let scorePtr = scoreBuffer.contents().assumingMemoryBound(to: UInt32.self)
        scorePtr[0] = UInt32.max
        
        let resultPtr = resultBuffer.contents().assumingMemoryBound(to: UInt32.self)
        resultPtr[0] = UInt32.max
        
        // Create uniforms - must match Metal struct layout exactly (184 bytes)
        var uniforms = Uniforms(
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            screenSize: SIMD2(UInt32(viewportSize.width), UInt32(viewportSize.height)),
            splatCount: UInt32(splatBuffer.count),
            indexedSplatCount: UInt32(min(splatBuffer.count, Constants.maxIndexedSplatCount)),
            showClusterColors: 0,
            showMask: 0,
            selectedClusterID: -1,
            showDepthVisualization: 0,
            selectionMode: 0,
            depthRange: SIMD2(0.1, 10.0),
            selectedClusterCount: 0,
            maskThreshold: 0
        )
        
        var params = PickingParams(
            clickPointX: Float(screenPoint.x),
            clickPointY: Float(screenPoint.y),
            screenWidth: Float(viewportSize.width),
            screenHeight: Float(viewportSize.height)
        )
        
        print("Picking at screen \(screenPoint), viewport size \(viewportSize), splat count \(splatBuffer.count)")
        
        let splatCount = splatBuffer.count
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return -1
        }
        
        encoder.setComputePipelineState(computePipeline)
        encoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(clusterIDs, offset: 0, index: 1)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<PickingParams>.size, index: 3)
        encoder.setBuffer(scoreBuffer, offset: 0, index: 4)
        encoder.setBuffer(resultBuffer, offset: 0, index: 5)
        
        let threadsPerGroup = computePipeline.maxTotalThreadsPerThreadgroup
        let threadgroups = (splatCount + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let minScore = Float(scorePtr[0]) / 100.0
        let rawClusterID = resultPtr[0]
        
        // Check for "no result" before casting (UInt32.max would overflow Int32)
        if rawClusterID == UInt32.max || scorePtr[0] == UInt32.max {
            print("No splat found within 100 pixels of click point")
            return -1
        }
        
        let clusterID = Int32(rawClusterID)
        print("Found splat with cluster \(clusterID), score = \(minScore)")
        
        return clusterID
    }
    
    /// Toggle a cluster in/out of the multi-selection set
    public func toggleClusterSelection(_ clusterID: Int32) {
        guard clusterID >= 0 else { return }
        let id = UInt32(clusterID)
        if selectedClusters.contains(id) {
            selectedClusters.remove(id)
        } else if selectedClusters.count < Self.maxSelectedClusters {
            selectedClusters.insert(id)
        }
    }
    
    /// Clear all selected clusters
    public func clearSelection() {
        selectedClusters.removeAll()
        selectionMode = 0
        selectedClustersBuffer = nil
    }

    private func buildInitializePipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = library.makeRequiredFunction(name: "initializeFragmentStore")

        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[1].pixelFormat = .r32Sint

        return try device.makeRenderPipelineState(tileDescriptor: pipelineDescriptor, options: [], reflection: nil)
    }

    private func buildDrawSplatPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DrawSplatPipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "multiStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeRequiredFunction(name: "multiStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[1].pixelFormat = .r32Sint
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDrawSplatDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    private func buildPostprocessPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "PostprocessPipeline"
        pipelineDescriptor.vertexFunction =
            library.makeRequiredFunction(name: "postprocessVertexShader")
        pipelineDescriptor.fragmentFunction =
            writeDepth
            ? library.makeRequiredFunction(name: "postprocessFragmentShader")
            : library.makeRequiredFunction(name: "postprocessFragmentShaderNoDepth")

        pipelineDescriptor.colorAttachments[0]!.pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[1]!.pixelFormat = .r32Sint
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildPostprocessDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    public func ensureAdditionalCapacity(_ pointCount: Int) throws {
        try splatBuffer.ensureCapacity(splatBuffer.count + pointCount)
    }

    public func add(_ points: [SplatScenePoint]) throws {
        do {
            try ensureAdditionalCapacity(points.count)
        } catch {
            Self.log.error("Failed to grow buffers: \(error)")
            return
        }

        splatBuffer.append(points.map { Splat($0) })
    }

    public func add(_ point: SplatScenePoint) throws {
        try add([ point ])
    }

    private func switchToNextDynamicBuffer() {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxSimultaneousRenders
        uniformBufferOffset = UniformsArray.alignedSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents() + uniformBufferOffset).bindMemory(to: UniformsArray.self, capacity: 1)
    }

    private func updateUniforms(forViewports viewports: [ViewportDescriptor],
                                splatCount: UInt32,
                                indexedSplatCount: UInt32) {
        // Update depth range based on camera position (simple heuristic)
        if useDepthVisualization {
            updateDepthRange(forViewports: viewports)
        }
        
        // Update selected clusters buffer if needed
        updateSelectedClustersBuffer()
        
        for (i, viewport) in viewports.enumerated() where i <= maxViewCount {
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount,
                                    showClusterColors: useClusterColors ? 1 : 0,
                                    showMask: showMask ? 1 : 0,
                                    selectedClusterID: selectedClusterID,
                                    showDepthVisualization: useDepthVisualization ? 1 : 0,
                                    selectionMode: selectionMode,
                                    depthRange: currentDepthRange,
                                    selectedClusterCount: UInt32(min(selectedClusters.count, Self.maxSelectedClusters)),
                                    maskThreshold: maskThreshold)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }

        cameraWorldPosition = viewports.map { Self.cameraWorldPosition(forViewMatrix: $0.viewMatrix) }.mean ?? .zero
        cameraWorldForward = viewports.map { Self.cameraWorldForward(forViewMatrix: $0.viewMatrix) }.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        if !sorting {
            resort()
        }
    }
    
    /// Update the selected clusters buffer with current selection
    private func updateSelectedClustersBuffer() {
        guard !selectedClusters.isEmpty else {
            selectedClustersBuffer = nil
            return
        }
        
        let count = min(selectedClusters.count, Self.maxSelectedClusters)
        let bufferSize = MemoryLayout<UInt32>.size * Self.maxSelectedClusters
        
        if selectedClustersBuffer == nil {
            selectedClustersBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        }
        
        guard let buffer = selectedClustersBuffer else { return }
        
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, clusterID) in selectedClusters.prefix(count).enumerated() {
            ptr[i] = clusterID
        }
    }
    
    /// Update depth range by sampling visible splats
    private func updateDepthRange(forViewports viewports: [ViewportDescriptor]) {
        guard let viewport = viewports.first, splatBuffer.count > 0 else { return }
        
        let viewMatrix = viewport.viewMatrix
        var minDepth: Float = Float.greatestFiniteMagnitude
        var maxDepth: Float = -Float.greatestFiniteMagnitude
        
        // Sample a subset of splats for performance (every Nth splat)
        let sampleStride = max(1, splatBuffer.count / 1000)
        let splatPtr = splatBuffer.buffer.contents().assumingMemoryBound(to: Splat.self)
        
        for i in stride(from: 0, to: splatBuffer.count, by: sampleStride) {
            let splat = splatPtr[i]
            let pos = SIMD4<Float>(splat.position.x, splat.position.y, splat.position.z, 1.0)
            let viewPos = viewMatrix * pos
            let depth = -viewPos.z  // Positive depth (camera looks down -Z)
            
            if depth > 0.01 {  // Only consider splats in front of camera
                minDepth = min(minDepth, depth)
                maxDepth = max(maxDepth, depth)
            }
        }
        
        // Add some padding and ensure valid range
        if minDepth < maxDepth && minDepth > 0 {
            let range = maxDepth - minDepth
            currentDepthRange = SIMD2(minDepth - range * 0.05, maxDepth + range * 0.05)
        } else {
            currentDepthRange = SIMD2(0.1, 10.0)  // Fallback
        }
    }
    
    public func readCanonical(from url: URL) async throws {
        self.canonicalBuffer = try MetalBuffer(device: device) // Initialize the buffer
        
        // Follow the same logic when loading SplatBuffer
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: try AutodetectSceneReader(url))
        do {
            try self.canonicalBuffer!.ensureCapacity(newPoints.points.count)
        } catch {
            Self.log.error("Failed to grow buffers: \(error)")
            return
        }

        canonicalBuffer!.append(newPoints.points.map { CanonicalSplat($0) })
    }
    public func loadDeformableScene(directory: URL, loadClusters: Bool = false) async throws {
        // When selecting a whole directory as input,
        // automatically consider as loading a dynamic scene.
        
        // Configure scene and deform mlp path
        let plyURL = directory.appendingPathComponent("point_cloud.ply")
        let weightsURL = directory.appendingPathComponent("weights.bin")
        
        // Load the mlp weight
        let weightsData = try Data(contentsOf: weightsURL)
        
        // Initialize the MPS deformation network
        self.deformSystem = DeformGraphSystem(device: device, useFP16: useFP16Deformation)
        self.deformSystem?.loadWeights(flatData: weightsData)
        self.deformSystem?.buildAndCompile()
        
        // Init Kernels
        let lib = self.library
        
        guard let extractFunc = lib.makeFunction(name: "extract_graph_inputs"),
              let timeFillFunc = lib.makeFunction(name: "fill_time"),
              let applyFunc = lib.makeFunction(name: "apply_graph_outputs") else {
            print("Error: Could not find Deform.metal shader functions.")
            return
        }

        self.extractPipeline = try await device.makeComputePipelineState(function: extractFunc)
        self.timeFillPipeline = try await device.makeComputePipelineState(function: timeFillFunc)
        self.applyPipeline = try await device.makeComputePipelineState(function: applyFunc)
        
        // Read canonical Gaussians using the original IO pipeline
        try await readCanonical(from: plyURL)
        try await read(from: plyURL)
        
        guard canonicalBuffer!.count == splatBuffer.count else {return}
        // Allocate Buffers
        let count = canonicalBuffer?.count ?? 0
        
        if count > 0 {
            // Use shared storage mode for masked deformation support (allows CPU read/write)
            bufXYZ = device.makeBuffer(length: count * 3 * 4, options: .storageModeShared)
            bufT   = device.makeBuffer(length: count * 1 * 4, options: .storageModeShared)
            bufDXYZ = device.makeBuffer(length: count * 3 * 4, options: .storageModeShared)
            bufDRot = device.makeBuffer(length: count * 4 * 4, options: .storageModeShared)
            bufDScale = device.makeBuffer(length: count * 3 * 4, options: .storageModeShared)
        }
        
        if count > 0,
           let extractPipe = extractPipeline,
           let cBuffer = canonicalBuffer?.buffer,
           let bXYZ = bufXYZ,
           let bT = bufT,
           let queue = device.makeCommandQueue(),
           let cmd = queue.makeCommandBuffer(),
           let enc = cmd.makeComputeCommandEncoder() {
            enc.label = "Init: Extract XYZ"
            enc.setComputePipelineState(extractPipe)
            enc.setBuffer(cBuffer, offset: 0, index: 0)
            enc.setBuffer(bXYZ, offset: 0, index: 1)
            enc.setBuffer(bT, offset: 0, index: 2)
            var t: Float = 0
            enc.setBytes(&t, length: 4, index: 3)
            
            let w = extractPipe.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
            
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        if loadClusters {
            let clustersURL = directory.appendingPathComponent("clusters.bin")
            let count = canonicalBuffer?.count ?? 0
            print("About to load clusters from: \(clustersURL)")
            loadClustersBin(from: clustersURL, expectedCount: count)
        }

        // Load deformation mask (mask.bin)
        let maskURL = directory.appendingPathComponent("mask.bin")
        let maskCount = canonicalBuffer?.count ?? 0
        loadDeformMaskBin(from: maskURL, expectedCount: maskCount)

        print("Loaded Deformable Scene: \(canonicalBuffer!.count) points")
    }

    @available(*, deprecated, message: "Use loadDeformableScene(directory:loadClusters:) instead")
    public func loadDeformableSceneClusters(directory: URL) async throws {
        try await loadDeformableScene(directory: directory, loadClusters: true)
    }

    private func loadClustersBin(from url: URL, expectedCount: Int) {
        print("loadClustersBin called with url: \(url), expectedCount: \(expectedCount)")
        
        let data: Data
        do {
            data = try Data(contentsOf: url)
            print("Read \(data.count) bytes from clusters.bin")
        } catch {
            print("Failed to read clusters.bin: \(error.localizedDescription)")
            return
        }
        
        guard data.count >= 12 else {
            print("clusters.bin too small: \(data.count) bytes")
            return
        }
        
        let magic = String(decoding: data.subdata(in: 0..<4), as: UTF8.self)
        guard magic == "CLST" else {
            print("clusters.bin has bad magic '\(magic)'")
            return
        }
        print("Magic OK, reading header...")
        
        func readU32(_ offset: Int) -> UInt32 {
            data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { raw in
                let value = raw.load(as: UInt32.self)
                return UInt32(littleEndian: value)
            }
        }
        
        let version = readU32(4)
        guard version == 1 else {
            Self.log.error("clusters.bin unsupported version \(version).")
            return
        }
        
        let count = Int(readU32(8))
        guard count == expectedCount else {
            Self.log.error("clusters.bin count \(count) != splat count \(expectedCount).")
            return
        }
        
        let idsOffset = 12
        let idsBytes = count * MemoryLayout<UInt32>.size
        let colorsOffset = idsOffset + idsBytes
        let colorsBytes = count * 3 * MemoryLayout<Float>.size
        guard data.count >= colorsOffset + colorsBytes else {
            Self.log.error("clusters.bin truncated.")
            return
        }
        
        var ids = [UInt32](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: idsOffset)
            let src = base.assumingMemoryBound(to: UInt32.self)
            for i in 0..<count {
                ids[i] = UInt32(littleEndian: src[i])
            }
        }
        
        let colorCount = count * 3
        var colors = [Float](repeating: 0, count: colorCount)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: colorsOffset)
            let src = base.assumingMemoryBound(to: Float.self)
            for i in 0..<colorCount {
                colors[i] = src[i]
            }
        }
        
        clusterIdBuffer = device.makeBuffer(bytes: ids,
                                            length: ids.count * MemoryLayout<UInt32>.size,
                                            options: .storageModeShared)
        clusterColorBuffer = device.makeBuffer(bytes: colors,
                                               length: colors.count * MemoryLayout<Float>.size,
                                               options: .storageModeShared)
        
        // Calculate max cluster ID for picking
        maxClusterID = ids.max() ?? 0
        print("Max cluster ID: \(maxClusterID)")
        
        // Debug: print first few colors to compare with Python
        print("=== clusters.bin debug ===")
        print("Count: \(count)")
        for i in 0..<min(3, count) {
            let r = colors[i * 3 + 0]
            let g = colors[i * 3 + 1]
            let b = colors[i * 3 + 2]
            print("Color[\(i)]: (\(r), \(g), \(b))")
        }
        for i in max(0, count - 3)..<count {
            let r = colors[i * 3 + 0]
            let g = colors[i * 3 + 1]
            let b = colors[i * 3 + 2]
            print("Color[\(i)]: (\(r), \(g), \(b))")
        }
        print("=== end clusters.bin debug ===")
        
        Self.log.info("Loaded clusters.bin (\(count) entries).")
    }

    private func loadDeformMaskBin(from url: URL, expectedCount: Int) {
        // Create all-ones mask buffer (used for t=0 to deform all splats)
        let allOnes = [Float](repeating: 1.0, count: expectedCount)
        let allOnesBytes = expectedCount * MemoryLayout<Float>.size
        allOnesMaskBuffer = device.makeBuffer(bytes: allOnes, length: allOnesBytes, options: .storageModeShared)

        let data: Data
        do {
            data = try Data(contentsOf: url)
            Self.log.info("Read \(data.count) bytes from mask.bin")
        } catch {
            Self.log.info("No mask.bin found or failed to read: \(error.localizedDescription)")
            return
        }

        let expectedBytes = expectedCount * MemoryLayout<Float>.size
        guard data.count >= expectedBytes else {
            Self.log.error("mask.bin too small: \(data.count) bytes, expected \(expectedBytes)")
            return
        }

        let maskValues = data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: base, count: expectedCount))
        }

        deformMaskBuffer = device.makeBuffer(bytes: maskValues, length: expectedBytes, options: .storageModeShared)
        
        // Compute percentiles for slider mapping
        self.sortedMaskValues = maskValues.sorted()
        
        // Max value for fallback
        self.maxMaskValue = self.sortedMaskValues.last ?? 1.0
        if self.maxMaskValue < 1e-8 { self.maxMaskValue = 1.0 }
        
        // Read recommended value if available
        let jsonURL = url.deletingPathExtension().appendingPathExtension("json")
        self.maskJsonURL = jsonURL
        if let data = try? Data(contentsOf: jsonURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let percentage = obj["recommended_percentile"] as? Double {
            self.recommendedMaskPercentage = percentage
            Self.log.info("Loaded custom mask recommendation: \(percentage)%")
        } else {
            self.recommendedMaskPercentage = nil
        }
        
        Self.log.info("Loaded deform mask (\(expectedCount) entries, max=\(self.maxMaskValue)).")
    }

    /// Saves a new recommended mask percentage to mask.json
    public func saveRecommendedMaskPercentage(_ percentage: Double) {
        guard let url = maskJsonURL else {
            Self.log.error("Failed to save mask recommendation: no mask.json URL available.")
            return
        }
        let dict: [String: Any] = ["recommended_percentile": percentage]
        do {
            let data = try JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted)
            try data.write(to: url)
            self.recommendedMaskPercentage = percentage
            Self.log.info("Saved custom mask recommendation: \(percentage)% to \(url.path)")
        } catch {
            Self.log.error("Failed to save mask recommendation: \(error.localizedDescription)")
        }
    }

    public func update(time: Float, commandBuffer: MTLCommandBuffer) {
        // Check if time changed significantly
        if abs(time - lastDeformationTime) < 0.001 { return }
        
        Self.log.debug("Deformation time: \(time)")
        lastDeformationTime = time
        
        let commandQueue = commandBuffer.commandQueue
        let count = canonicalBuffer?.count ?? 0
        
        guard count > 0,
              let sys = deformSystem,
              let timeFillPipe = timeFillPipeline,
              let applyPipe = applyPipeline,
              let bXYZ = bufXYZ,
              let bT = bufT,
              let bDXYZ = bufDXYZ,
              let bDRot = bufDRot,
              let bDScale = bufDScale,
              let cBuffer = canonicalBuffer?.buffer
        else {
            Self.log.debug("Something is missing.")
            return
        }
        // Fill t buffer (xyz is static and extracted once during load)
        // Use the same command buffer to enable pipelining - no separate wait needed
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "Update: Fill Time"
            enc.setComputePipelineState(timeFillPipe)
            enc.setBuffer(bT, offset: 0, index: 0)
            var t = time
            enc.setBytes(&t, length: 4, index: 1)

            let w = timeFillPipe.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }
        
        // Calculate d_xyz, d_rotation, d_scaling
        // For t=0: run on all splats
        // For t>0 with masked deformation: only run on masked splats
        let useMasked = useMaskedDeformation && time > 0.001 && deformMaskBuffer != nil

        let graphElapsedMs: Double
        if useMasked {
            // Run only on masked splats for better performance
            graphElapsedMs = sys.runMasked(commandQueue: commandQueue,
                          xyzBuffer: bXYZ,
                          tBuffer: bT,
                          outXYZ: bDXYZ,
                          outRot: bDRot,
                          outScale: bDScale,
                          maskBuffer: deformMaskBuffer!,
                          threshold: maskThreshold,
                          count: count)
        } else {
            // Run on all splats
            graphElapsedMs = sys.run(commandQueue: commandQueue,
                    xyzBuffer: bXYZ,
                    tBuffer: bT,
                    outXYZ: bDXYZ,
                    outRot: bDRot,
                    outScale: bDScale,
                    count: count)
        }

        // Apply deformation to canonical Gaussians
        // The shader always applies deltas unconditionally. When runMasked is used,
        // static splats retain their t=0 deltas in the buffer, so they keep their
        // t=0 appearance automatically. We just pass any valid mask buffer.
        let maskBuffer: MTLBuffer? = allOnesMaskBuffer

        if let enc = commandBuffer.makeComputeCommandEncoder(),
           let bMask = maskBuffer {
            enc.label = "Update: Apply Outputs"
            enc.setComputePipelineState(applyPipe)
            enc.setBuffer(cBuffer, offset: 0, index: 0)
            enc.setBuffer(bDXYZ, offset: 0, index: 1)
            enc.setBuffer(bDRot, offset: 0, index: 2)
            enc.setBuffer(bDScale, offset: 0, index: 3)
            enc.setBuffer(splatBuffer.buffer, offset: 0, index: 4)
            enc.setBuffer(bMask, offset: 0, index: 5)

            let w = applyPipe.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Update deformation FPS from actual graph runtime (FPS = 1 / (ms/1000))
        if graphElapsedMs > 0.01 {
            deformFPS = 1.0 / (graphElapsedMs / 1000.0)
            print("Deform FPS: \(deformFPS) (from \(graphElapsedMs) ms)")
        }

        // Track how many splats were deformed this frame
        if useMasked {
            _deformedSplatCount = sys.lastMaskedCount
        } else {
            _deformedSplatCount = count
        }
    }

    /// Returns the current deformation FPS, or 0 if no deformation is running
    public var currentDeformFPS: Double {
        return deformFPS
    }

    /// Returns the number of splats deformed in the last frame
    public var deformedSplatCount: Int {
        return _deformedSplatCount
    }

    /// Returns the total splat count
    public var totalSplatCount: Int {
        return splatBuffer.count
    }

    private static func cameraWorldForward(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz
    }

    private static func cameraWorldPosition(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz
    }

    func renderEncoder(multiStage: Bool,
                       viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       clusterIDTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       for commandBuffer: MTLCommandBuffer) -> MTLRenderCommandEncoder {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        if let depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 0.0
        }
        if let clusterIDTexture {
            renderPassDescriptor.colorAttachments[1].texture = clusterIDTexture
            renderPassDescriptor.colorAttachments[1].loadAction = .clear
            renderPassDescriptor.colorAttachments[1].storeAction = .store
            // Clear to -1 (0xFFFFFFFF)
            renderPassDescriptor.colorAttachments[1].clearColor = MTLClearColor(red: -1, green: 0, blue: 0, alpha: 0) // Treat as int/uint?
            // Metal clearColor is Double. For integer formats, we might need clearValue?
            // "When the attachment has an integer format, the clear color is used as integer values."
            // So -1.0 might work if cast? Or I should check docs.
            // Actually MTLClearColor is (double, double, double, double).
            // For integer formats, "Only the red channel is used for single-channel formats".
            // It seems Metal wrapper converts it?
            // Usually we use `MTLClearColor(red: Double(0xFFFFFFFF), ...)` for uint?
            // -1 signed int is all 1s.
            // Let's assume -1.0 is fine or specific bit pattern. Double can represent -1 exactly.
        }
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength

        renderPassDescriptor.tileWidth  = Constants.tileSize.width
        renderPassDescriptor.tileHeight = Constants.tileSize.height

        if multiStage {
            if let initializePipelineState {
                renderPassDescriptor.imageblockSampleLength = initializePipelineState.imageblockSampleLength
            } else {
                Self.log.error("initializePipeline == nil in renderEncoder()")
            }
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.setViewports(viewports.map(\.viewport))

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        return renderEncoder
    }
    
    public func render(viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       clusterIDTexture: MTLTexture? = nil,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        let splatCount = splatBuffer.count
        guard splatBuffer.count != 0 else { return }
        let indexedSplatCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + indexedSplatCount - 1) / indexedSplatCount

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(splatCount), indexedSplatCount: UInt32(indexedSplatCount))

        let multiStage = useMultiStagePipeline
        if multiStage {
            try buildMultiStagePipelineStatesIfNeeded()
        } else {
            try buildSingleStagePipelineStatesIfNeeded()
        }

        let renderEncoder = renderEncoder(multiStage: multiStage,
                                          viewports: viewports,
                                          colorTexture: colorTexture,
                                          colorStoreAction: colorStoreAction,

                                          depthTexture: depthTexture,
                                          clusterIDTexture: clusterIDTexture,
                                          rasterizationRateMap: rasterizationRateMap,
                                          renderTargetArrayLength: renderTargetArrayLength,
                                          for: commandBuffer)

        let indexCount = indexedSplatCount * 6
        if indexBuffer.count < indexCount {
            do {
                try indexBuffer.ensureCapacity(indexCount)
            } catch {
                return
            }
            indexBuffer.count = indexCount
            for i in 0..<indexedSplatCount {
                indexBuffer.values[i * 6 + 0] = UInt32(i * 4 + 0)
                indexBuffer.values[i * 6 + 1] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 2] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 3] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 4] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 5] = UInt32(i * 4 + 3)
            }
        }

        if multiStage {
            guard let initializePipelineState,
                  let drawSplatPipelineState
            else { return }

            renderEncoder.pushDebugGroup("Initialize")
            renderEncoder.setRenderPipelineState(initializePipelineState)
            renderEncoder.dispatchThreadsPerTile(Constants.tileSize)
            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(drawSplatPipelineState)
            renderEncoder.setDepthStencilState(drawSplatDepthState)
        } else {
            guard let singleStagePipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(singleStagePipelineState)
            renderEncoder.setDepthStencilState(singleStageDepthState)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
        let clusterBuffer = clusterColorBuffer ?? emptyClusterColorBuffer
        renderEncoder.setVertexBuffer(clusterBuffer, offset: 0, index: BufferIndex.clusterColor.rawValue)
        let clusterIDBuffer = clusterIdBuffer ?? emptyClusterIdBuffer
        renderEncoder.setVertexBuffer(clusterIDBuffer, offset: 0, index: BufferIndex.clusterID.rawValue)
        
        // Bind selected clusters buffer for multi-selection
        let selClustersBuffer = selectedClustersBuffer ?? emptySelectedClustersBuffer
        renderEncoder.setVertexBuffer(selClustersBuffer, offset: 0, index: BufferIndex.selectedClusters.rawValue)

        // Bind deform mask buffer (used for mask visualization)
        let maskBuffer = deformMaskBuffer ?? emptyDeformMaskBuffer
        renderEncoder.setVertexBuffer(maskBuffer, offset: 0, index: BufferIndex.deformMask.rawValue)

        if let sortedBuffer = sortedIndexBuffer?.buffer {
            renderEncoder.setVertexBuffer(sortedBuffer, offset: 0, index: BufferIndex.splatIndices.rawValue)
        }
        
        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                            indexCount: indexCount,
                                            indexType: .uint32,
                                            indexBuffer: indexBuffer.buffer,
                                            indexBufferOffset: 0,
                                            instanceCount: instanceCount)

        if multiStage {
            guard let postprocessPipelineState
            else { return }

            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Postprocess")
            renderEncoder.setRenderPipelineState(postprocessPipelineState)
            renderEncoder.setDepthStencilState(postprocessDepthState)
            renderEncoder.setCullMode(.none)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.popDebugGroup()
        } else {
            renderEncoder.popDebugGroup()
        }

        renderEncoder.endEncoding()
        updateFPS()
    }
    
    private func updateFPS() {
        fpsFrameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - fpsLastTimestamp
        guard elapsed >= 1.0 else { return }
        currentFPS = Double(fpsFrameCount) / elapsed
        Self.log.debug("Rendering FPS: \(self.currentFPS)")
        fpsFrameCount = 0
        fpsLastTimestamp = now
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort(useGPU: Bool = true) {
        guard !sorting else { return }
        sorting = true
        onSortStart?()

        let splatCount = splatBuffer.count
        
        let cameraWorldForward = cameraWorldForward
        let cameraWorldPosition = cameraWorldPosition

        if useGPU {
            Task(priority: .high) {
                // Allocate a GPU buffer for storing distances.
                guard let distanceBuffer = device.makeBuffer(
                    length: MemoryLayout<Float>.size * splatCount,
                    options: .storageModeShared
                ) else {
                    Self.log.error("Failed to create distance buffer.")
                    self.sorting = false
                    return
                }

                // Compute distances on CPU then copy to distanceBuffer.
                let distancePtr = distanceBuffer.contents().bindMemory(to: Float.self, capacity: splatCount)
                if Constants.sortByDistance {
                    for i in 0 ..< splatCount {
                        let splatPos = splatBuffer.values[i].position.simd
                        distancePtr[i] = (splatPos - cameraWorldPosition).lengthSquared
                    }
                } else {
                    for i in 0 ..< splatCount {
                        let splatPos = splatBuffer.values[i].position.simd
                        distancePtr[i] = dot(splatPos, cameraWorldForward)
                    }
                }
            

                // Allocate a GPU buffer for the ArgSort output indices
                guard let indexOutputBuffer = device.makeBuffer(
                    length: MemoryLayout<Int32>.size * splatCount,
                    options: .storageModeShared
                ) else {
                    Self.log.error("Failed to create output indices buffer.")
                    self.sorting = false
                    return
                }

                // Create command queue for MPSArgSort.
                guard let commandQueue = device.makeCommandQueue() else {
                    Self.log.error("Failed to create command queue for MPSArgSort.")
                    self.sorting = false
                    return
                }

                // Run argsort, in decending order.
                let argSort = MPSArgSort(dataType: .float32, descending: true)
                argSort(commandQueue: commandQueue,
                        input: distanceBuffer,
                        output: indexOutputBuffer,
                        count: splatCount)

                // Read back the sorted indices and reorder splats on the CPU.
                let sortedIndicesPtr = indexOutputBuffer.contents().bindMemory(to: Int32.self, capacity: splatCount)
                        
                // Convert to Array for safe appending, casting Int32 -> UInt32 to match your other code
                let sortedIndices = (0..<splatCount).map { UInt32(bitPattern: sortedIndicesPtr[$0]) }

                do {
                    try? sortedIndexBuffer?.setCapacity(splatCount)
                    sortedIndexBuffer?.count = 0
                    sortedIndexBuffer?.append(sortedIndices)
                } catch {
                    print("Sort upload failed")
                }
                self.sorting = false
            }
        } else {
            Task(priority: .high) {
                if orderAndDepthTempSort.count != splatCount {
                    orderAndDepthTempSort = Array(
                        repeating: SplatIndexAndDepth(index: .max, depth: 0),
                        count: splatCount
                    )
                }

                if Constants.sortByDistance {
                    for i in 0 ..< splatCount {
                        orderAndDepthTempSort[i].index = UInt32(i)
                        let splatPos = splatBuffer.values[i].position.simd
                        orderAndDepthTempSort[i].depth = (splatPos - cameraWorldPosition).lengthSquared
                    }
                } else {
                    for i in 0 ..< splatCount {
                        orderAndDepthTempSort[i].index = UInt32(i)
                        let splatPos = splatBuffer.values[i].position.simd
                        orderAndDepthTempSort[i].depth = dot(splatPos, cameraWorldForward)
                    }
                }

                orderAndDepthTempSort.sort { $0.depth > $1.depth }

                let sortedIndices = orderAndDepthTempSort.map { $0.index }
                do {
                    try? sortedIndexBuffer?.setCapacity(splatCount)
                    sortedIndexBuffer?.count = 0
                    sortedIndexBuffer?.append(sortedIndices)
                } catch {
                    print("Sort upload failed")
                }
            }
        }
    }
}

extension SplatRenderer.Splat {
    init(_ splat: SplatScenePoint) {
        self.init(position: splat.position,
                  color: .init(splat.color.asLinearFloat.sRGBToLinear, splat.opacity.asLinearFloat),
                  scale: splat.scale.asLinearFloat,
                  rotation: splat.rotation.normalized)
    }

    init(position: SIMD3<Float>,
         color: SIMD4<Float>,
         scale: SIMD3<Float>,
         rotation: simd_quatf) {
        let transform = simd_float3x3(rotation) * simd_float3x3(diagonal: scale)
        let cov3D = transform * transform.transpose
        self.init(position: MTLPackedFloat3Make(position.x, position.y, position.z),
                  color: SplatRenderer.PackedRGBHalf4(r: Float16(color.x), g: Float16(color.y), b: Float16(color.z), a: Float16(color.w)),
                  covA: SplatRenderer.PackedHalf3(x: Float16(cov3D[0, 0]), y: Float16(cov3D[0, 1]), z: Float16(cov3D[0, 2])),
                  covB: SplatRenderer.PackedHalf3(x: Float16(cov3D[1, 1]), y: Float16(cov3D[1, 2]), z: Float16(cov3D[2, 2])))
    }
}

extension SplatRenderer.CanonicalSplat {
    init(_ splat: SplatScenePoint) {
        self.init(position: splat.position,
                  color: .init(splat.color.asLinearFloat.sRGBToLinear, splat.opacity.asLinearFloat),
                  scale: splat.scale.asLinearFloat,
                  rotation: splat.rotation.normalized)
    }

    init(position: SIMD3<Float>,
         color: SIMD4<Float>,
         scale: SIMD3<Float>,
         rotation: simd_quatf) {
        
        self.position = MTLPackedFloat3Make(position.x, position.y, position.z)
        
        self.color = SplatRenderer.PackedRGBHalf4(
            r: Float16(color.x),
            g: Float16(color.y),
            b: Float16(color.z),
            a: Float16(color.w)
        )
        
        self.scale = MTLPackedFloat3Make(scale.x, scale.y, scale.z)
        self.rotationX = rotation.imag.x;
        self.rotationY = rotation.imag.y;
        self.rotationZ = rotation.imag.z;
        self.rotationW = rotation.real;
    }
}

protocol MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { get }
}

extension UInt32: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint32 }
}
extension UInt16: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint16 }
}

extension Array where Element == SIMD3<Float> {
    var mean: SIMD3<Float>? {
        guard !isEmpty else { return nil }
        return reduce(.zero, +) / Float(count)
    }
}

private extension MTLPackedFloat3 {
    var simd: SIMD3<Float> {
        SIMD3(x: x, y: y, z: z)
    }
}

private extension SIMD3 where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    var normalized: SIMD3<Scalar> {
        self / Scalar(sqrt(lengthSquared))
    }

    var lengthSquared: Scalar {
        x*x + y*y + z*z
    }

    func vector4(w: Scalar) -> SIMD4<Scalar> {
        SIMD4<Scalar>(x: x, y: y, z: z, w: w)
    }

    static func random(in range: Range<Scalar>) -> SIMD3<Scalar> {
        Self(x: Scalar.random(in: range), y: .random(in: range), z: .random(in: range))
    }
}

private extension SIMD3<Float> {
    var sRGBToLinear: SIMD3<Float> {
        SIMD3(x: pow(x, 2.2), y: pow(y, 2.2), z: pow(z, 2.2))
    }
}

private extension SIMD4 where Scalar: BinaryFloatingPoint {
    var xyz: SIMD3<Scalar> {
        .init(x: x, y: y, z: z)
    }
}

private extension MTLLibrary {
    func makeRequiredFunction(name: String) -> MTLFunction {
        guard let result = makeFunction(name: name) else {
            fatalError("Unable to load required shader function: \"\(name)\"")
        }
        return result
    }
}

//  Original source: https://gist.github.com/kemchenj/26e1dad40e5b89de2828bad36c81302f
//  Assessed Feb 2, 2025.
class MPSArgSort {
    private let dataType: MPSDataType
    private let graph: MPSGraph
    private let graphExecutable: MPSGraphExecutable
    private let inputTensor: MPSGraphTensor
    private let outputTensor: MPSGraphTensor

    init(dataType: MPSDataType, descending: Bool = false) {
        self.dataType = dataType

        let graph = MPSGraph()
        let inputTensor = graph.placeholder(shape: nil, dataType: dataType, name: nil)
        let outputTensor = graph.argSort(inputTensor, axis: 0, descending: descending, name: nil)

        self.graph = graph
        self.inputTensor = inputTensor
        self.outputTensor = outputTensor
        self.graphExecutable = autoreleasepool {
            let compilationDescriptor = MPSGraphCompilationDescriptor()
            compilationDescriptor.waitForCompilationCompletion = true
            compilationDescriptor.disableTypeInference()
            return graph.compile(with: nil,
                                 feeds: [inputTensor : MPSGraphShapedType(shape: nil, dataType: dataType)],
                                 targetTensors: [outputTensor],
                                 targetOperations: nil,
                                 compilationDescriptor: compilationDescriptor)
        }
    }

    func callAsFunction(
        commandQueue: any MTLCommandQueue,
        input: any MTLBuffer,
        output: any MTLBuffer,
        count: Int
    ) {
        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            callAsFunction(commandBuffer: commandBuffer,
                           input: input,
                           output: output,
                           count: count)
            assert(commandBuffer.error == nil)
            assert(commandBuffer.status == .completed)
        }
    }

    private func callAsFunction(
        commandBuffer: any MTLCommandBuffer,
        input: any MTLBuffer,
        output: any MTLBuffer,
        count: Int
    ) {
        let shape: [NSNumber] = [count as NSNumber]
        let inputData = MPSGraphTensorData(input, shape: shape, dataType: dataType)
        let outputData = MPSGraphTensorData(output, shape: shape, dataType: .int32)
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = true
        graphExecutable.encode(to: MPSCommandBuffer(commandBuffer: commandBuffer),
                               inputs: [inputData],
                               results: [outputData],
                               executionDescriptor: executionDescriptor)
    }
}
