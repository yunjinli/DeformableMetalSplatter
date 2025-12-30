#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import SwiftUI

class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)
    
    // Enable gestures
    // Drag
    var yaw: Float = 0.0
    var pitch: Float = 0.0
    // Zoom
    var cameraDistance: Float = Constants.modelCenterZ
    // Pan xy
    var panX: Float = 0.0
    var panY: Float = 0.0

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
            try await splat.read(from: url)
            modelRenderer = splat
        case .sampleBox:
            modelRenderer = try! await SampleBoxRenderer(device: device,
                                                         colorFormat: metalKitView.colorPixelFormat,
                                                         depthFormat: metalKitView.depthStencilPixelFormat,
                                                         sampleCount: metalKitView.sampleCount,
                                                         maxViewCount: 1,
                                                         maxSimultaneousRenders: Constants.maxSimultaneousRenders)
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
        guard let modelRenderer else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        do {
            try modelRenderer.render(viewports: [viewport],
                                     colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                     colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                     depthTexture: view.depthStencilTexture,
                                     rasterizationRateMap: nil,
                                     renderTargetArrayLength: 0,
                                     to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        commandBuffer.present(drawable)

        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }
}

#endif // os(iOS) || os(macOS)
