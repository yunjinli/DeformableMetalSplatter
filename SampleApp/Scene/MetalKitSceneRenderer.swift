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
    public var selectedClusterID: Int32 = -1  // -1 means show all
    public var showDepthVisualization: Bool = false
    // Total length of the video/animation in seconds
    var animationDuration: Double = 10.0
    
    var drawableSize: CGSize = .zero

    init?(_ metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try await SplatRenderer(device: device,
                                                colorFormat: metalKitView.colorPixelFormat,
                                                depthFormat: metalKitView.depthStencilPixelFormat,
                                                sampleCount: metalKitView.sampleCount,
                                                maxViewCount: 1,
                                                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
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
        // Turn common 3D GS PLY files rightside-up. This isn't generally meaningful, it just
        // happens to be a useful default for the most common datasets at the moment.
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * rotationMatrix * commonUpCalibration,
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
            splatRenderer.selectedClusterID = selectedClusterID
            splatRenderer.useDepthVisualization = showDepthVisualization
            splatRenderer.update(time: timeToPass, commandBuffer: commandBuffer)
        }

        // Rendering Step
        do {
            try modelRenderer?.render(viewports: [viewport],
                                     colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                     colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                     depthTexture: view.depthStencilTexture,
                                     rasterizationRateMap: nil,
                                     renderTargetArrayLength: 0,
                                     to: commandBuffer) // Pass the SAME buffer
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
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
        Self.log.debug("MTKView draw FPS: \(fps)")
        fpsFrameCount = 0
        fpsLastTimestamp = now
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
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
}

#endif // os(iOS) || os(macOS)
