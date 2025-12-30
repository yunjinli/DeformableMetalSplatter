#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif

struct MetalKitSceneView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?

    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        var startCameraDistance: Float = 0.0
        
        
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            
            let translation = gesture.translation(in: gesture.view)
            
            // Check how many fingers are touching the screen
            if gesture.numberOfTouches == 1 {
                // One finger: Orbit
                let sensitivity: Float = 0.01
                renderer.yaw += Float(translation.x) * sensitivity
                renderer.pitch += Float(translation.y) * sensitivity
                
            } else if gesture.numberOfTouches == 2 {
                // Two fingers: XY pan
                let panSensitivity: Float = 0.005
                renderer.panX += Float(translation.x) * panSensitivity
                renderer.panY -= Float(translation.y) * panSensitivity
            }
            
            // Reset translation so we get incremental updates
            gesture.setTranslation(.zero, in: gesture.view)
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let renderer = renderer else { return }

            if gesture.state == .began {
                // Store the current distance when we start pinching
                startCameraDistance = renderer.cameraDistance
            } else if gesture.state == .changed {
                // Calculate new distance.
                let newDistance = startCameraDistance / Float(gesture.scale)
                renderer.cameraDistance = min(max(newDistance, -20.0), -0.5)
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

#if os(macOS)
    func makeNSView(context: NSViewRepresentableContext<MetalKitSceneView>) -> MTKView {
        makeView(context.coordinator)
    }
#elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitSceneView>) -> MTKView {
        let metalKitView = makeView(context.coordinator)
        
        // Add Gesture Recognizer
        let panGesture = UIPanGestureRecognizer(target: context.coordinator,
                                                action: #selector(Coordinator.handlePan(_:)))
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator,
                                                action: #selector(Coordinator.handlePinch(_:)))
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 2
        metalKitView.addGestureRecognizer(panGesture)
        metalKitView.addGestureRecognizer(pinchGesture)
        
        return metalKitView
    }
#endif

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        coordinator.renderer = renderer
        metalKitView.delegate = renderer

        Task {
            do {
                try await renderer?.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }

        return metalKitView
    }

#if os(macOS)
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitSceneView>) {
        updateView(context.coordinator)
    }
#elseif os(iOS)
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitSceneView>) {
        updateView(context.coordinator)
    }
#endif

    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        Task {
            do {
                try await renderer.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

#endif // os(iOS) || os(macOS)
