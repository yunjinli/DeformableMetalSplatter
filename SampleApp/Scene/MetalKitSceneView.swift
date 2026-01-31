#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif


struct MetalKitSceneView: View {
    var modelIdentifier: ModelIdentifier?
    
    // State for the slider
    @State private var time: Float = 0.0
    @State private var isManualTime: Bool = true
    @State private var showClusterColors: Bool = false
    @State private var showDepthVisualization: Bool = false
    @State private var selectedClusterID: Int32 = -1  // -1 means show all
    @State private var showControls: Bool = false  // Controls hidden by default on phone
    @State private var coordinateMode: Int = 0  // 0=default, 1=Z-up→Y-up, 2=flip, 3=none
    
    private let coordinateModeLabels = ["Default", "Z→Y", "Y→Z", "None"]

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                MetalView(modelIdentifier: modelIdentifier,
                          manualTime: isManualTime ? time : nil,
                          showClusterColors: showClusterColors,
                          showDepthVisualization: showDepthVisualization,
                          selectedClusterID: $selectedClusterID,
                          coordinateMode: coordinateMode)
                    .ignoresSafeArea()

            // UI Overlay
            VStack(spacing: 8) {
                // Time slider with manual toggle inline + Show All on right
                HStack(spacing: 8) {
                    Toggle("Manual", isOn: $isManualTime)
                        .toggleStyle(.button)
                        .font(.caption)
                    
                    if isManualTime {
                        Text(String(format: "%.2f", time))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white)
                            .frame(width: 40, alignment: .trailing)
                        
                        Slider(value: $time, in: 0...1)
                            .accentColor(.blue)
                    } else {
                        Spacer()
                    }
                    
                    if selectedClusterID >= 0 {
                        Button("Show All") {
                            selectedClusterID = -1
                        }
                        .buttonStyle(.bordered)
                        .font(.caption)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .cornerRadius(8)
                
                // Collapsible controls section
                if showControls {
                    VStack(spacing: 8) {
                        HStack {
                            Toggle("Cluster Colors", isOn: Binding(
                                get: { showClusterColors },
                                set: { newValue in
                                    showClusterColors = newValue
                                    if newValue { showDepthVisualization = false }
                                }
                            ))
                                .toggleStyle(.button)
                                .font(.caption)
                            
                            Toggle("Depth", isOn: Binding(
                                get: { showDepthVisualization },
                                set: { newValue in
                                    showDepthVisualization = newValue
                                    if newValue { showClusterColors = false }
                                }
                            ))
                                .toggleStyle(.button)
                                .font(.caption)
                        }
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .cornerRadius(8)
                        
                        // Coordinate system picker
                        HStack {
                            Text("Axis:")
                                .font(.caption)
                                .foregroundStyle(.white)
                            
                            Picker("", selection: $coordinateMode) {
                                ForEach(0..<coordinateModeLabels.count, id: \.self) { index in
                                    Text(coordinateModeLabels[index]).tag(index)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 200)
                        }
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .cornerRadius(8)
                        
                        // Cluster selection info
                        HStack {
                            if selectedClusterID >= 0 {
                                Text("Cluster: \(selectedClusterID)")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.white)
                            } else {
                                Text("Tap splat to isolate cluster")
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.8))
                            }
                        }
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .cornerRadius(8)
                    }
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                
                // Toggle to show/hide extra controls
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showControls.toggle()
                    }
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: showControls ? "chevron.down" : "chevron.up")
                        Text(showControls ? "Hide Controls" : "More Controls")
                    }
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.8))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial)
                .cornerRadius(8)
            }
            .padding()
            .frame(maxWidth: 400)
            }
        }
    }
}

private struct MetalView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    var manualTime: Float?
    var showClusterColors: Bool?
    var showDepthVisualization: Bool?
    @Binding var selectedClusterID: Int32
    var coordinateMode: Int

    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        var startCameraDistance: Float = 0.0
        var selectedClusterIDBinding: Binding<Int32>?
        
#if os(iOS)
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard gesture.state == .ended,
                  let renderer = renderer,
                  let view = gesture.view else { return }

            let location = gesture.location(in: view)

            // Convert to drawable coordinates for pick.
            // Use actual drawable size vs view bounds for accurate scaling.
            let drawableSize = renderer.drawableSize
            let scaleX = drawableSize.width / view.bounds.width
            let scaleY = drawableSize.height / view.bounds.height

            // iOS touch coords have origin at top-left, which matches what the
            // picking shader expects. Do NOT flip Y here (shader already handles it).
            let screenPoint = CGPoint(
                x: location.x * scaleX,
                y: location.y * scaleY
            )

            if let clusterID = renderer.pickClusterAt(screenPoint) {
                selectedClusterIDBinding?.wrappedValue = clusterID
                renderer.selectedClusterID = clusterID
            }
        }

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
#endif // os(iOS)
    }

    func makeCoordinator() -> Coordinator {
        let coordinator = Coordinator()
        coordinator.selectedClusterIDBinding = $selectedClusterID
        return coordinator
    }

#if os(macOS)
    func makeNSView(context: Context) -> MTKView {
        // Use a custom subclass to capture mouse events
        let metalKitView = InteractiveMTKView()
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        context.coordinator.renderer = renderer
        metalKitView.delegate = renderer
        
        // Link the view back to the renderer for input handling
        metalKitView.renderer = renderer
        metalKitView.coordinator = context.coordinator

        loadModel(renderer)

        return metalKitView
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.renderer?.manualTime = manualTime
        context.coordinator.selectedClusterIDBinding = $selectedClusterID
        if let showClusterColors {
            context.coordinator.renderer?.showClusterColors = showClusterColors
        }
        if let showDepthVisualization {
            context.coordinator.renderer?.showDepthVisualization = showDepthVisualization
        }
        context.coordinator.renderer?.selectedClusterID = selectedClusterID
        context.coordinator.renderer?.coordinateMode = coordinateMode
        updateView(context.coordinator)
    }
    
    // Custom MTKView subclass to handle Mouse/Trackpad events
    class InteractiveMTKView: MTKView {
        weak var renderer: MetalKitSceneRenderer?
        weak var coordinator: Coordinator?
        
        override var acceptsFirstResponder: Bool { true }
        
        // Click: Pick cluster (works in any color mode)
        override func mouseDown(with event: NSEvent) {
            guard let renderer = renderer else { return }
            
            let location = convert(event.locationInWindow, from: nil)
            // Flip Y for Metal coordinate system
            let flippedY = bounds.height - location.y
            
            // Scale point from view coordinates (points) to drawable coordinates (pixels)
            // This is necessary for Retina displays where drawable size != bounds size
            let scale = window?.backingScaleFactor ?? 1.0
            let screenPoint = CGPoint(x: location.x * scale, y: flippedY * scale)
            
            if let clusterID = renderer.pickClusterAt(screenPoint) {
                coordinator?.selectedClusterIDBinding?.wrappedValue = clusterID
                renderer.selectedClusterID = clusterID
            }
        }
        
        // Orbit: Left Mouse Drag
        override func mouseDragged(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let sensitivity: Float = 0.01
            renderer.yaw += Float(event.deltaX) * sensitivity
            renderer.pitch += Float(event.deltaY) * sensitivity
        }
        
        // Pan: Right Mouse Drag (or Control + Click Drag)
        override func rightMouseDragged(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let panSensitivity: Float = 0.005
            renderer.panX += Float(event.deltaX) * panSensitivity
            renderer.panY -= Float(event.deltaY) * panSensitivity
        }
        
        // Pan alternative: Other Mouse Drag (Middle click)
        override func otherMouseDragged(with event: NSEvent) {
            rightMouseDragged(with: event)
        }
        
        // Zoom: Scroll Wheel
        override func scrollWheel(with event: NSEvent) {
            guard let renderer = renderer else { return }
            let scrollSensitivity: Float = 0.5
            // Note: deltaY is usually inverse to distance expectation on scroll
            renderer.cameraDistance += Float(event.deltaY) * scrollSensitivity
            renderer.cameraDistance = min(max(renderer.cameraDistance, -20.0), -0.5)
        }
        
        // Zoom: Pinch Gesture on Trackpad
        override func magnify(with event: NSEvent) {
            guard let renderer = renderer else { return }
            // Magnification is a scale factor (e.g. 1.0 + magnification)
            // We adjust distance inversely to scale
            let scale = Float(1.0 + event.magnification)
            if scale > 0 {
                renderer.cameraDistance /= scale
                renderer.cameraDistance = min(max(renderer.cameraDistance, -20.0), -0.5)
            }
        }
    }
#endif // os(macOS)

#if os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalView>) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        context.coordinator.renderer = renderer
        metalKitView.delegate = renderer

        // Add Gesture Recognizers
        let tapGesture = UITapGestureRecognizer(target: context.coordinator,
                                                action: #selector(Coordinator.handleTap(_:)))
        tapGesture.numberOfTouchesRequired = 1
        let twoFingerTapGesture = UITapGestureRecognizer(target: context.coordinator,
                                                         action: #selector(Coordinator.handleTap(_:)))
        twoFingerTapGesture.numberOfTouchesRequired = 2
        let panGesture = UIPanGestureRecognizer(target: context.coordinator,
                                                action: #selector(Coordinator.handlePan(_:)))
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator,
                                                    action: #selector(Coordinator.handlePinch(_:)))
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 2
        tapGesture.require(toFail: panGesture)
        twoFingerTapGesture.require(toFail: panGesture)
        metalKitView.addGestureRecognizer(tapGesture)
        metalKitView.addGestureRecognizer(twoFingerTapGesture)
        metalKitView.addGestureRecognizer(panGesture)
        metalKitView.addGestureRecognizer(pinchGesture)

        loadModel(renderer)
        
        return metalKitView
    }
    
    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.renderer?.manualTime = manualTime
        if let showClusterColors {
            context.coordinator.renderer?.showClusterColors = showClusterColors
        }
        if let showDepthVisualization {
            context.coordinator.renderer?.showDepthVisualization = showDepthVisualization
        }
        context.coordinator.renderer?.selectedClusterID = selectedClusterID
        context.coordinator.renderer?.coordinateMode = coordinateMode
        updateView(context.coordinator)
    }
#endif // os(iOS)

    private func loadModel(_ renderer: MetalKitSceneRenderer?) {
        Task {
            do {
                if let modelIdentifier = modelIdentifier {
                    try await renderer?.load(modelIdentifier)
                }
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
    
    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        Task {
            do {
                if let modelIdentifier = modelIdentifier {
                    try await renderer.load(modelIdentifier)
                }
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

#endif // os(iOS) || os(macOS)
