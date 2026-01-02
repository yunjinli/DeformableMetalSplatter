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

    private static let log =
        Logger(subsystem: Bundle.module.bundleIdentifier!,
               category: "SplatRenderer")
    
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
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels

        var splatCount: UInt32
        var indexedSplatCount: UInt32
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
    var canonicalBuffer: MetalBuffer<CanonicalSplat>?
    var canonicalSplatBufferPrime: MetalBuffer<CanonicalSplat>?
    var sortedIndexBuffer: MetalBuffer<UInt32>?
    var deformSystem: DeformGraphSystem?
    var extractPipeline: MTLComputePipelineState?
    var applyPipeline: MTLComputePipelineState?
    
    // Intermediate Buffers
    var bufXYZ: MTLBuffer?
    var bufT: MTLBuffer?
    var bufDXYZ: MTLBuffer?
    var bufDRot: MTLBuffer?
    var bufDScale: MTLBuffer?
    private var lastDeformationTime: Float = -1.0
    
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
//        self.canonicalSplatBufferPrime = try MetalBuffer(device: device)
        self.indexBuffer = try MetalBuffer(device: device)

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

    private func buildInitializePipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = library.makeRequiredFunction(name: "initializeFragmentStore")
        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

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
        for (i, viewport) in viewports.enumerated() where i <= maxViewCount {
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }

        cameraWorldPosition = viewports.map { Self.cameraWorldPosition(forViewMatrix: $0.viewMatrix) }.mean ?? .zero
        cameraWorldForward = viewports.map { Self.cameraWorldForward(forViewMatrix: $0.viewMatrix) }.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        if !sorting {
            resort()
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
    public func loadDeformableScene(directory: URL) async throws {
        // When selecting a whole directory as input,
        // automatically consider as loading a dynamic scene.
        
        // Configure scene and deform mlp path
        let plyURL = directory.appendingPathComponent("point_cloud.ply")
        let weightsURL = directory.appendingPathComponent("weights.bin")
        
        // Load the mlp weight
        let weightsData = try Data(contentsOf: weightsURL)
        
        // Initialize the MPS deformation network
        self.deformSystem = DeformGraphSystem(device: device)
        self.deformSystem?.loadWeights(flatData: weightsData)
        self.deformSystem?.buildAndCompile()
        
        // Init Kernels
        let lib = self.library
        
        guard let extractFunc = lib.makeFunction(name: "extract_graph_inputs"),
              let applyFunc = lib.makeFunction(name: "apply_graph_outputs") else {
            print("Error: Could not find Deform.metal shader functions.")
            return
        }

        self.extractPipeline = try await device.makeComputePipelineState(function: extractFunc)
        self.applyPipeline = try await device.makeComputePipelineState(function: applyFunc)
        
        // Read canonical Gaussians using the original IO pipeline
        try await readCanonical(from: plyURL)
        try await read(from: plyURL)
        
        guard canonicalBuffer!.count == splatBuffer.count else {return}
        // Allocate Buffers
        let count = canonicalBuffer?.count ?? 0
        
        if count > 0 {
            bufXYZ = device.makeBuffer(length: count * 3 * 4, options: .storageModePrivate)
            bufT   = device.makeBuffer(length: count * 1 * 4, options: .storageModePrivate)
            bufDXYZ = device.makeBuffer(length: count * 3 * 4, options: .storageModePrivate)
            bufDRot = device.makeBuffer(length: count * 4 * 4, options: .storageModePrivate)
            bufDScale = device.makeBuffer(length: count * 3 * 4, options: .storageModePrivate)
        }
        
//        guard let queue = device.makeCommandQueue(),
//              let sys = deformSystem,
//              let extractPipe = extractPipeline,
//              let applyPipe = applyPipeline,
//              let cBuffer = canonicalBuffer?.buffer,
//              let bXYZ = bufXYZ, let bT = bufT,
//              let bDXYZ = bufDXYZ, let bDRot = bufDRot, let bDScale = bufDScale else { return }
//
//        // Extract xyz from canonical Gaussians and send t=0.0
//        if let cmdA = queue.makeCommandBuffer(),
//           let enc = cmdA.makeComputeCommandEncoder() {
//            enc.label = "Init: Extract"
//            enc.setComputePipelineState(extractPipe)
//            enc.setBuffer(cBuffer, offset: 0, index: 0)
//            enc.setBuffer(bXYZ, offset: 0, index: 1)
//            enc.setBuffer(bT, offset: 0, index: 2)
//            var t = 0.0
//            enc.setBytes(&t, length: 4, index: 3)
//            let w = extractPipe.threadExecutionWidth
//            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
//                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
//            enc.endEncoding()
//            cmdA.commit()
//            cmdA.waitUntilCompleted()
//        }
//        
//        // Calculate d_xyz, d_rotation, d_scaling
//        sys.run(commandQueue: queue,
//                xyzBuffer: bXYZ,
//                tBuffer: bT,
//                outXYZ: bDXYZ,
//                outRot: bDRot,
//                outScale: bDScale,
//                count: count)
//        
//        // Apply deformation to canonical Gaussians
//        if let cmdC = queue.makeCommandBuffer(),
//           let enc = cmdC.makeComputeCommandEncoder() {
//            enc.label = "Init: Apply"
//            enc.setComputePipelineState(applyPipe)
//            enc.setBuffer(cBuffer, offset: 0, index: 0)
//            enc.setBuffer(bDXYZ, offset: 0, index: 1)
//            enc.setBuffer(bDRot, offset: 0, index: 2)
//            enc.setBuffer(bDScale, offset: 0, index: 3)
//            enc.setBuffer(splatBuffer.buffer, offset: 0, index: 4)
//            let w = applyPipe.threadExecutionWidth
//            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
//                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
//            enc.endEncoding()
//            cmdC.commit()
//            cmdC.waitUntilCompleted()
//        }
        print("Loaded Deformable Scene: \(canonicalBuffer!.count) points")
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
              let extractPipe = extractPipeline,
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
        // Extract xyz from canonical Gaussians and send t
        if let extractCmd = commandQueue.makeCommandBuffer(),
           let enc = extractCmd.makeComputeCommandEncoder() {
            
            enc.label = "Update: Extract Inputs"
            enc.setComputePipelineState(extractPipe)
            enc.setBuffer(cBuffer, offset: 0, index: 0)
            enc.setBuffer(bXYZ, offset: 0, index: 1)
            enc.setBuffer(bT, offset: 0, index: 2)
            var t = time
            enc.setBytes(&t, length: 4, index: 3)
            
            let w = extractPipe.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
            
            extractCmd.commit()
            extractCmd.waitUntilCompleted() // CPU Wait: Ensures data is ready for Graph
        }
        
        // Calculate d_xyz, d_rotation, d_scaling
        sys.run(commandQueue: commandQueue,
                xyzBuffer: bXYZ,
                tBuffer: bT,
                outXYZ: bDXYZ,
                outRot: bDRot,
                outScale: bDScale,
                count: count)
        
        // Apply deformation to canonical Gaussians
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "Update: Apply Outputs"
            enc.setComputePipelineState(applyPipe)
            enc.setBuffer(cBuffer, offset: 0, index: 0)
            enc.setBuffer(bDXYZ, offset: 0, index: 1)
            enc.setBuffer(bDRot, offset: 0, index: 2)
            enc.setBuffer(bDScale, offset: 0, index: 3)
            enc.setBuffer(splatBuffer.buffer, offset: 0, index: 4)
            
            let w = applyPipe.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }
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
        
        if let sortedBuffer = sortedIndexBuffer?.buffer {
            renderEncoder.setVertexBuffer(sortedBuffer, offset: 0, index: 2) // Index 2 = BufferIndex.splatIndices
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
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort(useGPU: Bool = true) {
        guard !sorting else { return }
        sorting = true
        onSortStart?()

        let splatCount = splatBuffer.count
        
        let cameraWorldForward = cameraWorldForward
        let cameraWorldPosition = cameraWorldPosition
        
//        // For benchmark.
//        guard splatCount > 0 else {
//            sorting = false
//            let elapsed: TimeInterval = 0
//            print("Sort time (\(useGPU ? "GPU" : "CPU")): \(elapsed) seconds")
//            onSortComplete?(elapsed)
//            return
//        }

        if useGPU {
            Task(priority: .high) {
//                let startTime = Date()

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
//        self.rotation = SIMD4<Float>(rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
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
