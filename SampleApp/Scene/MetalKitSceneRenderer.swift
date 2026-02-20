#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import os
import simd
import SwiftUI

class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "MetalKitSceneRenderer",
               category: "MetalKitSceneRenderer")
    private static let clipFeatureCacheLock = NSLock()
    private static var cachedCLIPFeaturesByModelPath: [String: (clusterIDs: [Int32], clusterFeatures: [[Float]])] = [:]

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)
    
    private var fpsFrameCount = 0
    private var fpsLastTimestamp = CFAbsoluteTimeGetCurrent()
    
    // Enable gestures
    // Drag
    var yaw: Float = 0.0
    var pitch: Float = 0.0
    // Zoom
    var cameraDistance: Float = Constants.modelCenterZ
    // Pan xy
    var panX: Float = 0.0
    var panY: Float = 0.0
    public var manualTime: Float? = nil
    public var showClusterColors: Bool = false
    public var showMask: Bool = false  // Show dynamic vs static splats
    public var selectedClusterID: Int32 = -1  // -1 means show all
    public var showDepthVisualization: Bool = false
    public var renderFPS: Double = 0.0  // Rendering FPS for display

    public var useMaskedCrops: Bool = true
    public var averageMaskedAndUnmasked: Bool = false  // Run both modes and average features
    public var useMaskedDeformation: Bool = false  // Use mask.bin for deformation (t > 0)
    public var maskThreshold: Float = 0.0  // Dynamic threshold for deformation mask

    // Multi-selection
    public var selectionMode: UInt32 = 0
    
    public var selectedClusterCount: Int {
        (modelRenderer as? SplatRenderer)?.selectedClusters.count ?? 0
    }
    
    public var selectedClusters: Set<UInt32> {
        (modelRenderer as? SplatRenderer)?.selectedClusters ?? []
    }
    
    func toggleClusterSelection(_ clusterID: Int32) {
        (modelRenderer as? SplatRenderer)?.toggleClusterSelection(clusterID)
    }
    
    func clearSelection() {
        selectionMode = 0
        (modelRenderer as? SplatRenderer)?.clearSelection()
    }
    
    /// Returns true if cluster data is available for this scene
    public var hasClusters: Bool {
        (modelRenderer as? SplatRenderer)?.hasClusters ?? false
    }
    /// Returns true if mask data is available for this scene
    public var hasMask: Bool {
        (modelRenderer as? SplatRenderer)?.hasMask ?? false
    }
    /// Returns the current deformation FPS (0 if no deformation)
    public var deformFPS: Double {
        (modelRenderer as? SplatRenderer)?.currentDeformFPS ?? 0.0
    }
    /// Returns the max mask value for slider scaling
    public var maxMaskValue: Float {
        (modelRenderer as? SplatRenderer)?.maxMaskValue ?? 1.0
    }
    
    /// Returns the recommended mask percentage (0-100) from the python generation script, if available
    public var recommendedMaskPercentage: Double? {
        (modelRenderer as? SplatRenderer)?.recommendedMaskPercentage
    }
    
    /// Saves the current mask percentage to mask.json
    public func saveRecommendedMaskPercentage(_ percentage: Double) {
        (modelRenderer as? SplatRenderer)?.saveRecommendedMaskPercentage(percentage)
    }
    
    /// Converts a percentage (0-100) to an absolute mask threshold using percentiles if available
    public func getMaskThreshold(forPercentage percentage: Double) -> Float {
        guard let splatRenderer = modelRenderer as? SplatRenderer,
              !splatRenderer.sortedMaskValues.isEmpty else {
            return Float(percentage / 100.0) * maxMaskValue
        }
        let sorted = splatRenderer.sortedMaskValues
        let clamped = max(0.0, min(100.0, percentage))
        let targetIndex = Int((clamped / 100.0) * Double(sorted.count - 1))
        return sorted[max(0, min(targetIndex, sorted.count - 1))]
    }
    
    /// Returns true if CoreML models are available for semantic clustering
    public var hasCLIPModels: Bool {
        clipService.hasImageEncoder && clipService.hasTextEncoder
    }
    // Coordinate system mode: 0=default, 1=rotate X -90°, 2=rotate X +90°, 3=no rotation
    public var coordinateMode: Int = 0
    // Total length of the video/animation in seconds
    var animationDuration: Double = 10.0
    
    var drawableSize: CGSize = .zero
    private var clusterIDTexture: MTLTexture?
    public var captureNextFrame: Bool = false

    // CLIP integration
    let clipService = CLIPService()
    /// True while encoding cluster crops in the background
    public var isEncodingClusters: Bool = false
    /// Encoding progress: (encoded, total)
    public var encodingProgress: (encoded: Int, total: Int) = (0, 0)
    /// Most recent query results: [(clusterID, similarity)]
    public var queryResults: [(clusterID: Int32, similarity: Float)] = []
    /// Callback invoked on main thread when encoding finishes
    public var onEncodingComplete: (() -> Void)?
    /// Callback invoked on main thread with status text during encoding
    public var onStatusUpdate: ((String) -> Void)?
    /// Callback invoked with deformation FPS each frame
    public var onDeformFPSUpdate: ((Double) -> Void)?
    /// Callback invoked with deformed/total splat counts each frame
    public var onDeformedSplatCountUpdate: ((Int, Int) -> Void)?

    init?(_ metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        metalKitView.framebufferOnly = false // Allow reading back texture
    }

    private func clipCacheKey(for model: ModelIdentifier?) -> String? {
        guard case .gaussianSplat(let url, _) = model else { return nil }
        return url.standardizedFileURL.path
    }

    private func cacheCurrentCLIPFeatures() {
        guard let key = clipCacheKey(for: model), clipService.encodedClusterCount > 0 else { return }
        let snapshot = clipService.featuresSnapshot()
        Self.clipFeatureCacheLock.lock()
        Self.cachedCLIPFeaturesByModelPath[key] = snapshot
        Self.clipFeatureCacheLock.unlock()
        print("[CLIP-DEBUG] Cached renderer CLIP features for key=\(key), clusters=\(snapshot.clusterIDs.count)")
    }

    private func restoreCachedCLIPFeaturesIfAvailable() {
        guard clipService.encodedClusterCount == 0,
              let key = clipCacheKey(for: model) else { return }
        Self.clipFeatureCacheLock.lock()
        let snapshot = Self.cachedCLIPFeaturesByModelPath[key]
        Self.clipFeatureCacheLock.unlock()
        guard let snapshot,
              !snapshot.clusterIDs.isEmpty,
              !snapshot.clusterFeatures.isEmpty else { return }
        clipService.replaceFeatures(clusterIDs: snapshot.clusterIDs,
                                    clusterFeatures: snapshot.clusterFeatures)
        print("[CLIP-DEBUG] Restored renderer CLIP feature cache for key=\(key), clusters=\(snapshot.clusterIDs.count)")
    }

    /// Restores cached CLIP features (if needed) and returns available encoded cluster count.
    @discardableResult
    func ensureCLIPFeaturesReady() -> Int {
        restoreCachedCLIPFeaturesIfAvailable()
        return clipService.encodedClusterCount
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url, let useFP16):
            let splat = try SplatRenderer(device: device,
                                          colorFormat: metalKitView.colorPixelFormat,
                                          depthFormat: metalKitView.depthStencilPixelFormat,
                                          sampleCount: metalKitView.sampleCount,
                                          maxViewCount: 1,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            // Apply the precision setting before loading
            splat.useFP16Deformation = useFP16
            
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                // It is a directory: Assume Deformable Scene (ply + weights + clusters)
                try await splat.loadDeformableSceneClusters(directory: url)
            } else {
                // It is a file: Standard Static Scene
                try await splat.read(from: url)
            }
            
            modelRenderer = splat
        case .none:
            break
        }

        restoreCachedCLIPFeaturesIfAvailable()
    }

    private var viewport: ModelRendererViewportDescriptor {
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(Constants.fovy.radians),
                                                             aspectRatio: Float(drawableSize.width / drawableSize.height),
                                                             nearZ: 0.1,
                                                             farZ: 100.0)

        let rotationMatrixY = matrix4x4_rotation(radians: yaw, axis: SIMD3<Float>(0, 1, 0))
        let rotationMatrixX = matrix4x4_rotation(radians: pitch, axis: SIMD3<Float>(1, 0, 0))
        let rotationMatrix = rotationMatrixX * rotationMatrixY
        
        let translationMatrix = matrix4x4_translation(panX, panY, cameraDistance)
        
        // Coordinate system calibration - different modes for different scene conventions
        let coordinateCalibration: simd_float4x4
        switch coordinateMode {
        case 1:
            // Z-up to Y-up: rotate -90° around X
            coordinateCalibration = matrix4x4_rotation(radians: -.pi/2, axis: SIMD3<Float>(1, 0, 0))
        case 2:
            // Flip Z-up: rotate +90° around X
            coordinateCalibration = matrix4x4_rotation(radians: .pi/2, axis: SIMD3<Float>(1, 0, 0))
        case 3:
            // No rotation (identity)
            coordinateCalibration = matrix_identity_float4x4
        default:
            // Default: 180° around Z (original behavior for common 3DGS PLY files)
            coordinateCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))
        }

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * rotationMatrix * coordinateCalibration,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    func draw(in view: MTKView) {
        updateFPS()
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }
        commandBuffer.label = "Frame Command Buffer" // Name it for debugging
        
        let semaphore = self.inFlightSemaphore

        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }
        
        guard let drawable = view.currentDrawable else {
            commandBuffer.commit()
            return
        }

        // Deformation step
        let timeToPass: Float
        let now = Date().timeIntervalSinceReferenceDate
        if let manualTime = manualTime {
            timeToPass = manualTime
        } else {
            let loopedTime = now.remainder(dividingBy: animationDuration)
            timeToPass = Float((loopedTime < 0 ? loopedTime + animationDuration : loopedTime) / animationDuration)
        }
        
        if let splatRenderer = modelRenderer as? SplatRenderer {
            splatRenderer.useClusterColors = showClusterColors
            splatRenderer.showMask = showMask
            splatRenderer.selectedClusterID = selectedClusterID
            splatRenderer.useDepthVisualization = showDepthVisualization
            splatRenderer.selectionMode = selectionMode
            splatRenderer.useMaskedDeformation = useMaskedDeformation
            splatRenderer.maskThreshold = maskThreshold
            splatRenderer.update(time: timeToPass, commandBuffer: commandBuffer)
            
            // Push deformation FPS and splat counts to UI every frame
            let fps = splatRenderer.currentDeformFPS
            let deformedCount = splatRenderer.deformedSplatCount
            let totalCount = splatRenderer.totalSplatCount
            DispatchQueue.main.async { [weak self] in
                self?.onDeformFPSUpdate?(fps)
                self?.onDeformedSplatCountUpdate?(deformedCount, totalCount)
            }
        }

        // Rendering Step
        
        let modelViewports = [viewport]
        
        do {
            if let splatRenderer = modelRenderer as? SplatRenderer {
                // Convert viewport for SplatRenderer
                let splatViewports = modelViewports.map { mv in
                    SplatRenderer.ViewportDescriptor(viewport: mv.viewport,
                                                     projectionMatrix: mv.projectionMatrix,
                                                     viewMatrix: mv.viewMatrix,
                                                     screenSize: mv.screenSize)
                }
                
                try splatRenderer.render(viewports: splatViewports,
                                         colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                         colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                         depthTexture: view.depthStencilTexture,
                                         clusterIDTexture: clusterIDTexture,
                                         rasterizationRateMap: nil,
                                         renderTargetArrayLength: 0,
                                         to: commandBuffer)
            } else {
                try modelRenderer?.render(viewports: modelViewports,
                                          colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                          colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                          depthTexture: view.depthStencilTexture,
                                          rasterizationRateMap: nil,
                                          renderTargetArrayLength: 0,
                                          to: commandBuffer)
            }
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        if captureNextFrame {
            captureNextFrame = false
            isEncodingClusters = true
            print("[CLIP-DEBUG] captureNextFrame triggered, scheduling GPU completion handler")
            print("[CLIP-DEBUG] drawable.texture: \(drawable.texture.width)x\(drawable.texture.height) format=\(drawable.texture.pixelFormat.rawValue)")
            print("[CLIP-DEBUG] clusterIDTexture: \(String(describing: self.clusterIDTexture?.width))x\(String(describing: self.clusterIDTexture?.height))")
            print("[CLIP-DEBUG] clipService.imageEncoder loaded: \(self.clipService.hasImageEncoder)")
            
            // Immediately update status
            DispatchQueue.main.async { [weak self] in
                self?.onStatusUpdate?("Capturing frame…")
            }
            
            // Capture the cluster texture reference now (drawable may not survive completion handler)
            guard let clusterTexture = self.clusterIDTexture else {
                print("[CLIP-DEBUG] ERROR: clusterIDTexture is nil, aborting capture")
                isEncodingClusters = false
                DispatchQueue.main.async { [weak self] in
                    self?.onStatusUpdate?("Error: no cluster texture")
                }
                return
            }
            let rgbTexture = drawable.texture
            
            commandBuffer.addCompletedHandler { [weak self] _ in
                guard let self = self else {
                    print("[CLIP-DEBUG] ERROR: self is nil in completion handler")
                    return
                }
                print("[CLIP-DEBUG] GPU command buffer completed, starting cluster encoding on background thread")
                print("[CLIP-DEBUG] rgbTexture: \(rgbTexture.width)x\(rgbTexture.height)")
                print("[CLIP-DEBUG] clusterTexture: \(clusterTexture.width)x\(clusterTexture.height)")
                
                DispatchQueue.main.async {
                    self.onStatusUpdate?("Frame captured, processing…")
                }
                
                // Dispatch encoding to a background queue so it doesn't block the Metal callback
                let masked = self.useMaskedCrops
                let averageBoth = self.averageMaskedAndUnmasked
                DispatchQueue.global(qos: .userInitiated).async {
                    print("[CLIP-DEBUG] Background encoding started (masked=\(masked), averageBoth=\(averageBoth))")
                    self.clipService.encodeClusterCrops(rgb: rgbTexture, clusters: clusterTexture, useMaskedCrops: masked, averageMaskedAndUnmasked: averageBoth)
                    print("[CLIP-DEBUG] Background encoding finished, hasFeatures=\(self.clipService.hasFeatures)")
                    DispatchQueue.main.async {
                        self.cacheCurrentCLIPFeatures()
                        self.isEncodingClusters = false
                        self.onEncodingComplete?()
                        print("[CLIP-DEBUG] onEncodingComplete called on main thread")
                    }
                }
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func updateFPS() {
        fpsFrameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - fpsLastTimestamp
        guard elapsed >= 1.0 else { return }
        let fps = Double(fpsFrameCount) / elapsed
        renderFPS = fps
        Self.log.debug("MTKView draw FPS: \(fps)")
        fpsFrameCount = 0
        fpsLastTimestamp = now
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Sint,
                                                            width: Int(size.width),
                                                            height: Int(size.height),
                                                            mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .shared
        clusterIDTexture = device.makeTexture(descriptor: desc)
    }
    
    /// Pick the cluster at a screen position. Returns the cluster ID or nil if nothing was hit.
    func pickClusterAt(_ screenPoint: CGPoint) -> Int32? {
        guard let splatRenderer = modelRenderer as? SplatRenderer else {
            return nil
        }
        
        // Use the same viewport as rendering
        let viewportDesc = self.viewport
        
        let clusterID = splatRenderer.pickCluster(
            at: screenPoint,
            viewportSize: drawableSize,
            viewport: SplatRenderer.ViewportDescriptor(
                viewport: viewportDesc.viewport,
                projectionMatrix: viewportDesc.projectionMatrix,
                viewMatrix: viewportDesc.viewMatrix,
                screenSize: viewportDesc.screenSize
            )
        )
        
        return clusterID >= 0 ? clusterID : nil
    }

    
    // Helper to save frame
    /// Query the CLIP features with a text prompt and update `queryResults`.
    /// Returns the top-K cluster IDs sorted by descending cosine similarity.
    func queryText(_ text: String, topK: Int = 1) -> Set<UInt32> {
        _ = ensureCLIPFeaturesReady()
        let results = clipService.query(text: text)
        queryResults = results
        
        // Return top-K cluster IDs
        let selectedIDs = results.prefix(topK).map { UInt32($0.clusterID) }
        return Set(selectedIDs)
    }

    func saveFrame(rgb: MTLTexture, clusters: MTLTexture) {
        let timestamp = Int(Date().timeIntervalSince1970)
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let rgbUrl = docDir.appendingPathComponent("capture_\(timestamp)_rgb.raw")
        let clusterUrl = docDir.appendingPathComponent("capture_\(timestamp)_ids.bin")
        let metaUrl = docDir.appendingPathComponent("capture_\(timestamp)_meta.txt")
        
        let width = rgb.width
        let height = rgb.height
        
        // Save RGB (BGRA8)
        let bytesPerPixel = 4
        let rowBytes = width * bytesPerPixel
        var rgbData = Data(count: height * rowBytes)
        rgbData.withUnsafeMutableBytes { ptr in
            rgb.getBytes(ptr.baseAddress!, bytesPerRow: rowBytes, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        
        // Save Clusters (R32Sint)
        let clusterRowBytes = width * 4
        var clusterData = Data(count: height * clusterRowBytes)
        clusterData.withUnsafeMutableBytes { ptr in
            clusters.getBytes(ptr.baseAddress!, bytesPerRow: clusterRowBytes, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        
        do {
            try rgbData.write(to: rgbUrl)
            try clusterData.write(to: clusterUrl)
            
            let meta = "Width: \(width)\nHeight: \(height)\nRGB Format: BGRA8\nCluster Format: R32Sint\n"
            try meta.write(to: metaUrl, atomically: true, encoding: .utf8)
            
            print("Captured frame to:\n\(rgbUrl.path)\n\(clusterUrl.path)")
        } catch {
            print("Failed to save capture: \(error)")
        }
    }
}

#endif // os(iOS) || os(macOS)
