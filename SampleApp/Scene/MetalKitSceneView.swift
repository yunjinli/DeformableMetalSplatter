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
    
    init(modelIdentifier: ModelIdentifier?) {
        self.modelIdentifier = modelIdentifier
    }
    
    // State for the slider
    @State private var time: Float = 0.0
    @State private var isManualTime: Bool = true
    @State private var showClusterColors: Bool = false
    @State private var showMask: Bool = false  // Show dynamic vs static mask
    @State private var showDepthVisualization: Bool = false
    @State private var selectedClusterID: Int32 = -1  // -1 means show all
    @State private var showControls: Bool = false  // Controls hidden by default on phone
    @State private var coordinateMode: Int = 0  // 0=default, 1=Z-up→Y-up, 2=flip, 3=none
    @State private var hasClusters: Bool = false  // Whether clusters.bin was loaded
    @State private var hasMask: Bool = false  // Whether mask.bin was loaded
    @State private var hasCLIPModels: Bool = false  // Whether CoreML models are available
    
    // Multi-selection mode
    @State private var isSelectingMode: Bool = false  // Whether in multi-cluster selection mode
    @State private var selectedClusterCount: Int = 0  // Number of selected clusters
    @State private var deleteSelected: Bool = false  // If true, hide selected clusters instead of showing only them
    
    // Capture / CLIP encoding
    @State private var captureRequest: Bool = false
    @State private var isEncodingClusters: Bool = false
    @State private var hasClipFeatures: Bool = false
    @State private var encodingProgressText: String = ""  // e.g. "3 / 15"
    @State private var useMaskedCrops: Bool = true  // Mask non-cluster pixels in crops
    @State private var averageMaskedAndUnmasked: Bool = false  // Run both modes and average
    
    // CLIP text query
    @State private var queryText: String = ""
    @State private var queryTopResult: String = ""
    @State private var searchRequest: Bool = false
    @State private var queryStatusText: String = ""
    @State private var queryTopK: Int = 1  // Number of top clusters to select
    
    // Deformation settings
    @State private var useMaskedDeformation: Bool = false  // Use mask.bin for deformation
    @State private var maskThreshold: Double = 50.0  // Percentage 0-100
    @State private var recommendedMaskPercentage: Double? = nil // From generator script
    @State private var saveMaskRequest: Bool = false
    @State private var deformFPS: Double = 0.0  // Deformation FPS from renderer
    @State private var renderFPS: Double = 0.0  // Rendering FPS from renderer
    @State private var deformedSplatCount: Int = 0
    @State private var totalSplatCount: Int = 0
    
    private let coordinateModeLabels = ["Default", "Z→Y", "Y→Z", "None"]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                // Background Metal view
                MetalView(modelIdentifier: modelIdentifier,
                          manualTime: isManualTime ? time : nil,
                          showClusterColors: showClusterColors,
                          showMask: showMask,
                          showDepthVisualization: showDepthVisualization,
                          selectedClusterID: $selectedClusterID,
                          coordinateMode: coordinateMode,
                          hasClusters: $hasClusters,
                          hasMask: $hasMask,
                          hasCLIPModels: $hasCLIPModels,
                          useMaskedCrops: useMaskedCrops,
                          averageMaskedAndUnmasked: averageMaskedAndUnmasked,
                          isSelectingMode: $isSelectingMode,
                          selectedClusterCount: $selectedClusterCount,
                          deleteSelected: $deleteSelected,
                          captureRequest: $captureRequest,
                          isEncodingClusters: $isEncodingClusters,
                          hasClipFeatures: $hasClipFeatures,
                          queryText: $queryText,
                          encodingProgressText: $encodingProgressText,
                          searchRequest: $searchRequest,
                          queryStatusText: $queryStatusText,
                          queryTopK: $queryTopK,
                          useMaskedDeformation: useMaskedDeformation,
                          maskThreshold: maskThreshold,
                          deformFPS: $deformFPS,
                          renderFPS: $renderFPS,
                          deformedSplatCount: $deformedSplatCount,
                          totalSplatCount: $totalSplatCount,
                          saveMaskRequest: $saveMaskRequest,
                          recommendedMaskPercentage: $recommendedMaskPercentage)
                .ignoresSafeArea()
                .onChange(of: hasClusters) { _, newValue in
                    // Auto-disable cluster colors if clusters become unavailable
                    if !newValue && showClusterColors {
                        showClusterColors = false
                    }
                }
                .onChange(of: recommendedMaskPercentage) { _, newValue in
                    if let newPercentage = newValue {
                        maskThreshold = newPercentage
                    }
                }
                .onChange(of: hasMask) { _, newValue in
                    if !newValue && useMaskedDeformation {
                        useMaskedDeformation = false
                    }
                }
                
                // Deformation FPS overlay in top right corner (only when deforming)
                if !isManualTime || time > 0.01 {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(String(format: "Deformation: %.1f FPS", deformFPS))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(4)
                        if totalSplatCount > 0 {
                            Text("\(deformedSplatCount) / \(totalSplatCount) splats")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.8))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(4)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                    .padding(.top, 60)
                    .padding(.trailing, 16)
                }
                
                // UI Overlay
                VStack(spacing: 8) {
                    // Time slider with manual toggle inline + Show All on right
                    HStack(spacing: 8) {
                        Toggle("Manual", isOn: $isManualTime)
                            .toggleStyle(.button)
                            .font(.caption)
                        
                        // Masked deformation toggle
                        Toggle("Mask Static", isOn: $useMaskedDeformation)
                            .toggleStyle(.button)
                            .font(.caption)
                            .tint(.orange)
                            .disabled(!hasMask)
                            .help(hasMask ? "Only deform moving splats" : "No mask available for this scene")
                            
                        if !hasMask {
                            HStack(spacing: 4) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                Text("No mask.bin")
                            }
                            .font(.caption)
                            .foregroundColor(.yellow)
                        }

                        if useMaskedDeformation {
                            Text(String(format: "%.0f%%", maskThreshold))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(.orange)
                                .frame(width: 44, alignment: .trailing)
                            Slider(value: $maskThreshold, in: 0...100)
                                .accentColor(.orange)
                                .frame(maxWidth: 120)
                            Button(action: {
                                saveMaskRequest = true
                            }) {
                                Image(systemName: "square.and.arrow.down")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.orange)
                            .help("Save current threshold to mask.json")
                        }

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
                        
                        if selectedClusterID >= 0 || selectedClusterCount > 0 {
                            Button("Show All") {
                                selectedClusterID = -1
                                selectedClusterCount = 0
                                isSelectingMode = false
                                deleteSelected = false
                                queryText = ""
                                queryTopResult = ""
                                queryStatusText = ""
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
                            VStack(spacing: 4) {
                                HStack {
                                    Toggle("Cluster Colors", isOn: Binding(
                                        get: { showClusterColors },
                                        set: { newValue in
                                            showClusterColors = newValue
                                            if newValue { showDepthVisualization = false; showMask = false }
                                        }
                                    ))
                                    .toggleStyle(.button)
                                    .font(.caption)
                                    .disabled(!hasClusters)

                                    Toggle("Show Dynamic Splats", isOn: Binding(
                                        get: { showMask },
                                        set: { newValue in
                                            showMask = newValue
                                            if newValue { showDepthVisualization = false; showClusterColors = false }
                                        }
                                    ))
                                    .toggleStyle(.button)
                                    .font(.caption)
                                    .disabled(!hasMask)

                                    Toggle("Depth", isOn: Binding(
                                        get: { showDepthVisualization },
                                        set: { newValue in
                                            showDepthVisualization = newValue
                                            if newValue { showClusterColors = false; showMask = false }
                                        }
                                    ))
                                    .toggleStyle(.button)
                                    .font(.caption)
                                }
                                
                                // Warning when clusters.bin is not available
                                if !hasClusters {
                                    HStack(spacing: 4) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .foregroundStyle(.yellow)
                                        Text("clusters.bin not found")
                                            .foregroundStyle(.white.opacity(0.8))
                                    }
                                    .font(.caption2)
                                }
                                
                                // Warning when CoreML models are not available
                                if !hasCLIPModels {
                                    HStack(spacing: 4) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .foregroundStyle(.orange)
                                        Text("CoreML models not found - semantic search unavailable")
                                            .foregroundStyle(.white.opacity(0.8))
                                    }
                                    .font(.caption2)
                                }
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
                            
                            // Multi-cluster selection controls
                            HStack(spacing: 8) {
                                if isSelectingMode {
                                    // In selection mode
                                    Text("\(selectedClusterCount) selected")
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundStyle(.white)
                                    
                                    Button("Confirm") {
                                        isSelectingMode = false
                                        // Mode will be set to 2 (confirmed) in MetalView
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(.green)
                                    .font(.caption)
                                    
                                    Button("Delete") {
                                        isSelectingMode = false
                                        // Mode will be set to 3 (delete/hide) in MetalView
                                        deleteSelected = true
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(.orange)
                                    .font(.caption)
                                    
                                    Button("Cancel") {
                                        isSelectingMode = false
                                        selectedClusterCount = 0
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.red)
                                    .font(.caption)
                                } else if selectedClusterCount > 0 {
                                    // Confirmed/deleted selection active
                                    Text("\(deleteSelected ? "Hiding" : "Showing") \(selectedClusterCount) clusters")
                                        .font(.caption)
                                        .foregroundStyle(.white.opacity(0.8))
                                    
                                    Button("Edit") {
                                        isSelectingMode = true
                                        deleteSelected = false
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.blue)
                                    .font(.caption)
                                    
                                } else if selectedClusterID >= 0 {
                                    // Single cluster selection (legacy)
                                    Text("Cluster: \(selectedClusterID)")
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundStyle(.white)
                                } else {
                                    // No selection
                                    Button("Object Selection") {
                                        isSelectingMode = true
                                        selectedClusterID = -1  // Clear single selection
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.blue)
                                    .font(.caption)
                                    .disabled(!hasClusters)
                                }
                            }
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .cornerRadius(8)
                            
                            // Encode Clusters with CLIP
                            HStack(spacing: 6) {
                                Toggle("Mask", isOn: $useMaskedCrops)
                                    .toggleStyle(.button)
                                    .font(.caption)
                                    .disabled(isEncodingClusters)
                                    .help("Mask out non-cluster pixels before CLIP encoding")
                                
                                // Note: averageMaskedAndUnmasked can be enabled programmatically if needed
                                
                                Button(action: {
                                    captureRequest = true
                                    isEncodingClusters = true
                                    encodingProgressText = "Starting…"
                                }) {
                                    HStack(spacing: 4) {
                                        if isEncodingClusters {
                                            ProgressView()
                                                .scaleEffect(0.7)
                                        } else {
                                            Image(systemName: hasClipFeatures ? "checkmark.circle.fill" : "brain")
                                        }
                                        if isEncodingClusters && !encodingProgressText.isEmpty {
                                            Text(encodingProgressText)
                                        } else if hasClipFeatures {
                                            Text("Re-encode Clusters")
                                        } else {
                                            Text("Encode Clusters (CLIP)")
                                        }
                                    }
                                }
                                .buttonStyle(.bordered)
                                .tint(hasClipFeatures ? .green : .blue)
                                .disabled(isEncodingClusters || !hasClusters || !hasCLIPModels)
                            }
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .cornerRadius(8)
                            
                            // CLIP text query (always visible, enabled after encoding)
                            VStack(spacing: 6) {
                                HStack(spacing: 6) {
                                    Image(systemName: "magnifyingglass")
                                        .foregroundStyle(.white.opacity(0.6))
                                    TextField("Search clusters...", text: $queryText)
                                        .textFieldStyle(.plain)
                                        .foregroundStyle(.white)
                                        .disabled(!hasCLIPModels)
                                        .onSubmit {
                                            searchRequest = true
                                        }
                                    
                                    Button("Search") {
                                        searchRequest = true
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(.blue)
                                    .font(.caption)
                                    .disabled(queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !hasCLIPModels)
                                    
                                    if !queryText.isEmpty {
                                        Button(action: {
                                            queryText = ""
                                            selectedClusterID = -1
                                            selectedClusterCount = 0
                                            isSelectingMode = false
                                            queryTopResult = ""
                                            queryStatusText = ""
                                        }) {
                                            Image(systemName: "xmark.circle.fill")
                                                .foregroundStyle(.white.opacity(0.6))
                                        }
                                        .buttonStyle(.plain)
                                    }
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(.ultraThinMaterial)
                                .cornerRadius(8)
                                
                                // Top-K selector
                                HStack(spacing: 6) {
                                    Text("Top K")
                                        .font(.caption2)
                                        .foregroundStyle(.white.opacity(0.6))
                                    Button(action: { if queryTopK > 1 { queryTopK -= 1 } }) {
                                        Image(systemName: "minus.circle")
                                            .foregroundStyle(.white.opacity(queryTopK > 1 ? 0.8 : 0.3))
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(queryTopK <= 1)
                                    Text("\(queryTopK)")
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.white)
                                        .frame(width: 24, alignment: .center)
                                    Button(action: { queryTopK += 1 }) {
                                        Image(systemName: "plus.circle")
                                            .foregroundStyle(.white.opacity(0.8))
                                    }
                                    .buttonStyle(.plain)
                                }
                                .padding(.horizontal, 10)
                                
                                if !queryStatusText.isEmpty {
                                    Text(queryStatusText)
                                        .font(.caption2)
                                        .foregroundStyle(.yellow.opacity(0.9))
                                }
                                
                                if !queryTopResult.isEmpty {
                                    Text(queryTopResult)
                                        .font(.caption2)
                                        .foregroundStyle(.white.opacity(0.7))
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
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .padding()
                
                // Prominent encoding status overlay (center of screen)
                if isEncodingClusters {
                    VStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text(encodingProgressText.isEmpty ? "Starting…" : encodingProgressText)
                            .font(.system(.headline, design: .rounded))
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .animation(.easeInOut(duration: 0.15), value: encodingProgressText)
                    }
                    .padding(.horizontal, 32)
                    .padding(.vertical, 24)
                    .background(.ultraThinMaterial)
                    .background(Color.black.opacity(0.4))
                    .cornerRadius(16)
                    .shadow(color: .black.opacity(0.3), radius: 10, y: 4)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .transition(.opacity.combined(with: .scale(scale: 0.9)))
                    .animation(.easeInOut(duration: 0.25), value: isEncodingClusters)
                }
            }
        }
    }
    
    private struct MetalView: ViewRepresentable {
        var modelIdentifier: ModelIdentifier?
        var manualTime: Float?
        var showClusterColors: Bool?
        var showMask: Bool?
        var showDepthVisualization: Bool?
        @Binding var selectedClusterID: Int32
        var coordinateMode: Int
        @Binding var hasClusters: Bool
        @Binding var hasMask: Bool
        @Binding var hasCLIPModels: Bool
        var useMaskedCrops: Bool
        var averageMaskedAndUnmasked: Bool
        @Binding var isSelectingMode: Bool
        @Binding var selectedClusterCount: Int
        @Binding var deleteSelected: Bool
        @Binding var captureRequest: Bool
        @Binding var isEncodingClusters: Bool
        @Binding var hasClipFeatures: Bool
        @Binding var queryText: String
        @Binding var encodingProgressText: String
        @Binding var searchRequest: Bool
        @Binding var queryStatusText: String
        @Binding var queryTopK: Int
        var useMaskedDeformation: Bool
        var maskThreshold: Double
        @Binding var deformFPS: Double
        @Binding var renderFPS: Double
        @Binding var deformedSplatCount: Int
        @Binding var totalSplatCount: Int
        @Binding var saveMaskRequest: Bool
        @Binding var recommendedMaskPercentage: Double?
        
        class Coordinator: NSObject {
            var renderer: MetalKitSceneRenderer?
            var startCameraDistance: Float = 0.0
            var selectedClusterIDBinding: Binding<Int32>?
            var hasClustersBinding: Binding<Bool>?
            var hasMaskBinding: Binding<Bool>?
            var hasCLIPModelsBinding: Binding<Bool>?
            var isSelectingModeBinding: Binding<Bool>?
            var selectedClusterCountBinding: Binding<Int>?
            var isEncodingClustersBinding: Binding<Bool>?
            var hasClipFeaturesBinding: Binding<Bool>?
            var queryTextBinding: Binding<String>?
            var encodingProgressTextBinding: Binding<String>?
            var searchRequestBinding: Binding<Bool>?
            var queryStatusTextBinding: Binding<String>?
            var queryTopKBinding: Binding<Int>?
            var deformFPSBinding: Binding<Double>?
            var renderFPSBinding: Binding<Double>?
            var deformedSplatCountBinding: Binding<Int>?
            var totalSplatCountBinding: Binding<Int>?
            var recommendedMaskPercentageBinding: Binding<Double?>?
            /// Tracks last query text to avoid redundant queries
            var lastQueryText: String = ""
            /// Cached CLIP features to survive renderer/view lifecycle resets
            var cachedClusterIDs: [Int32] = []
            var cachedClusterFeatures: [[Float]] = []
            
            func processQuery() {
                guard let text = queryTextBinding?.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines) else { return }
                
                guard !text.isEmpty else {
                    queryStatusTextBinding?.wrappedValue = "Enter a search term first"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                        self?.queryStatusTextBinding?.wrappedValue = ""
                    }
                    return
                }
                
                // detailed CLIP debug logging
                guard let renderer = renderer else {
                    print("[CLIP-DEBUG] Search failed: Coordinator.renderer is nil")
                    queryStatusTextBinding?.wrappedValue = "⚠️ Internal Error: Renderer detached"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                        self?.queryStatusTextBinding?.wrappedValue = ""
                    }
                    return
                }
                
                if renderer.clipService.encodedClusterCount == 0,
                   !cachedClusterIDs.isEmpty,
                   !cachedClusterFeatures.isEmpty {
                    renderer.clipService.replaceFeatures(clusterIDs: cachedClusterIDs,
                                                         clusterFeatures: cachedClusterFeatures)
                    print("[CLIP-DEBUG] Restored cached CLIP features into renderer before query: \(cachedClusterIDs.count) clusters")
                }
                
                let encodedClusterCount = renderer.ensureCLIPFeaturesReady()
                print("[CLIP-DEBUG] processQuery: renderer=\(ObjectIdentifier(renderer)), encodedClusterCount=\(encodedClusterCount), hasFeatures=\(renderer.clipService.hasFeatures)")
                
                guard encodedClusterCount > 0 else {
                    print("[CLIP-DEBUG] Search failed: encodedClusterCount is 0. Clusters not encoded yet or encoding produced no features.")
                    queryStatusTextBinding?.wrappedValue = "⚠️ Encode clusters first before searching"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                        self?.queryStatusTextBinding?.wrappedValue = ""
                    }
                    return
                }
                
                queryStatusTextBinding?.wrappedValue = "Searching…"
                
                let topK = queryTopKBinding?.wrappedValue ?? 1
                let selectedIDs = renderer.queryText(text, topK: topK)
                
                renderer.clearSelection()
                for id in selectedIDs {
                    renderer.toggleClusterSelection(Int32(id))
                }
                // Enter selection mode so user can review/adjust before confirm or delete
                renderer.selectionMode = 1
                
                selectedClusterCountBinding?.wrappedValue = selectedIDs.count
                isSelectingModeBinding?.wrappedValue = true
                
                if selectedIDs.isEmpty {
                    queryStatusTextBinding?.wrappedValue = "No matches found"
                } else {
                    queryStatusTextBinding?.wrappedValue = "Found top \(selectedIDs.count) cluster(s)"
                }
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                    self?.queryStatusTextBinding?.wrappedValue = ""
                }
            }
            
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
                    if isSelectingModeBinding?.wrappedValue == true {
                        renderer.toggleClusterSelection(clusterID)
                        selectedClusterCountBinding?.wrappedValue = renderer.selectedClusterCount
                    }
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
            coordinator.hasClustersBinding = $hasClusters
            coordinator.hasMaskBinding = $hasMask
            coordinator.hasCLIPModelsBinding = $hasCLIPModels
            coordinator.isSelectingModeBinding = $isSelectingMode
            coordinator.selectedClusterCountBinding = $selectedClusterCount
            coordinator.isEncodingClustersBinding = $isEncodingClusters
            coordinator.hasClipFeaturesBinding = $hasClipFeatures
            coordinator.queryTextBinding = $queryText
            coordinator.encodingProgressTextBinding = $encodingProgressText
            coordinator.searchRequestBinding = $searchRequest
            coordinator.queryStatusTextBinding = $queryStatusText
            coordinator.queryTopKBinding = $queryTopK
            coordinator.deformFPSBinding = $deformFPS
            coordinator.renderFPSBinding = $renderFPS
            coordinator.deformedSplatCountBinding = $deformedSplatCount
            coordinator.totalSplatCountBinding = $totalSplatCount
            coordinator.recommendedMaskPercentageBinding = $recommendedMaskPercentage
            return coordinator
        }
        
        /// Shared logic to set up CLIP callbacks on the renderer
        private func setupCLIPCallbacks(renderer: MetalKitSceneRenderer, coordinator: Coordinator) {
            print("[CLIP-DEBUG] setupCLIPCallbacks called")
            renderer.onEncodingComplete = { [weak renderer] in
                let count = renderer?.clipService.encodedClusterCount ?? 0
                let hasEncoder = renderer?.clipService.hasImageEncoder ?? false
                let hasTextEnc = renderer?.clipService.hasTextEncoder ?? false
                print("[CLIP-DEBUG] onEncodingComplete: encodedCount=\(count), imgEnc=\(hasEncoder), txtEnc=\(hasTextEnc)")
                coordinator.isEncodingClustersBinding?.wrappedValue = false
                if let snapshot = renderer?.clipService.featuresSnapshot(), !snapshot.clusterIDs.isEmpty {
                    coordinator.cachedClusterIDs = snapshot.clusterIDs
                    coordinator.cachedClusterFeatures = snapshot.clusterFeatures
                    print("[CLIP-DEBUG] Cached CLIP feature snapshot: \(snapshot.clusterIDs.count) clusters")
                }
                let features = count > 0
                coordinator.hasClipFeaturesBinding?.wrappedValue = features
                print("[CLIP-DEBUG] set hasClipFeatures binding to \(features)")
                if count == 0 {
                    if !hasEncoder {
                        coordinator.encodingProgressTextBinding?.wrappedValue = "⚠️ Image encoder model missing from bundle"
                    } else {
                        coordinator.encodingProgressTextBinding?.wrappedValue = "⚠️ Encoding produced 0 features"
                    }
                    // Keep error visible longer
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                        coordinator.encodingProgressTextBinding?.wrappedValue = ""
                    }
                } else {
                    coordinator.encodingProgressTextBinding?.wrappedValue = "✓ \(count) clusters encoded"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        coordinator.encodingProgressTextBinding?.wrappedValue = ""
                    }
                }
            }
            
            // Renderer-level status updates (capture, GPU completion)
            renderer.onStatusUpdate = { statusText in
                print("[CLIP-DEBUG] renderer onStatusUpdate: \(statusText)")
                coordinator.encodingProgressTextBinding?.wrappedValue = statusText
            }
            
            // CLIPService-level status updates (reading textures, cropping, encoding x/y)
            renderer.clipService.onStatusUpdate = { statusText in
                print("[CLIP-DEBUG] clipService onStatusUpdate: \(statusText)")
                coordinator.encodingProgressTextBinding?.wrappedValue = statusText
            }
            
            // Deformation FPS updates (pushed every frame from draw loop)
            renderer.onDeformFPSUpdate = { fps in
                coordinator.deformFPSBinding?.wrappedValue = fps
            }
            renderer.onDeformedSplatCountUpdate = { deformed, total in
                coordinator.deformedSplatCountBinding?.wrappedValue = deformed
                coordinator.totalSplatCountBinding?.wrappedValue = total
            }
        }
        
#if os(macOS)
        func makeNSView(context: Context) -> MTKView {
            // Use a custom subclass to capture mouse events
            let metalKitView = InteractiveMTKView()
            
            if let metalDevice = MTLCreateSystemDefaultDevice() {
                metalKitView.device = metalDevice
            }
            
            guard let renderer = MetalKitSceneRenderer(metalKitView) else {
                return metalKitView
            }
            context.coordinator.renderer = renderer
            metalKitView.delegate = renderer
            
            // Link the view back to the renderer for input handling
            metalKitView.renderer = renderer
            metalKitView.coordinator = context.coordinator
            
            loadModel(renderer, coordinator: context.coordinator)
            setupCLIPCallbacks(renderer: renderer, coordinator: context.coordinator)
            
            return metalKitView
        }
        
        func updateNSView(_ view: MTKView, context: Context) {
            context.coordinator.renderer?.manualTime = manualTime
            context.coordinator.selectedClusterIDBinding = $selectedClusterID
            context.coordinator.hasClustersBinding = $hasClusters
            context.coordinator.hasCLIPModelsBinding = $hasCLIPModels
            context.coordinator.isSelectingModeBinding = $isSelectingMode
            context.coordinator.selectedClusterCountBinding = $selectedClusterCount
            context.coordinator.isEncodingClustersBinding = $isEncodingClusters
            context.coordinator.hasClipFeaturesBinding = $hasClipFeatures
            context.coordinator.queryTextBinding = $queryText
            context.coordinator.encodingProgressTextBinding = $encodingProgressText
            context.coordinator.searchRequestBinding = $searchRequest
            context.coordinator.queryStatusTextBinding = $queryStatusText
            context.coordinator.recommendedMaskPercentageBinding = $recommendedMaskPercentage
            
            if let showClusterColors {
                context.coordinator.renderer?.showClusterColors = showClusterColors
            }
            if let showMask {
                context.coordinator.renderer?.showMask = showMask
            }
            if let showDepthVisualization {
                context.coordinator.renderer?.showDepthVisualization = showDepthVisualization
            }
            context.coordinator.renderer?.selectedClusterID = selectedClusterID
            context.coordinator.renderer?.coordinateMode = coordinateMode

            context.coordinator.renderer?.useMaskedCrops = useMaskedCrops
            context.coordinator.renderer?.averageMaskedAndUnmasked = averageMaskedAndUnmasked
            context.coordinator.renderer?.useMaskedDeformation = useMaskedDeformation
            
            if let threshold = context.coordinator.renderer?.getMaskThreshold(forPercentage: maskThreshold) {
                context.coordinator.renderer?.maskThreshold = threshold
            }

            // Pass selection mode to renderer
            let mode: UInt32 = isSelectingMode ? 1 : (selectedClusterCount > 0 ? (deleteSelected ? 3 : 2) : 0)
            context.coordinator.renderer?.selectionMode = mode
            
            // Update hasClusters state from renderer and handle selection
            if let renderer = context.coordinator.renderer {
                if selectedClusterCount == 0 && !renderer.selectedClusters.isEmpty {
                    renderer.clearSelection()
                }
                let hasClustersNow = renderer.hasClusters
                let hasMaskNow = renderer.hasMask
                let hasFeaturesNow = renderer.clipService.hasFeatures
                let hasCLIPModelsNow = renderer.hasCLIPModels
                let deformFPSNow = renderer.deformFPS
                let renderFPSNow = renderer.renderFPS
                let recommendedNow = renderer.recommendedMaskPercentage
                DispatchQueue.main.async {
                    context.coordinator.hasClustersBinding?.wrappedValue = hasClustersNow
                    context.coordinator.hasMaskBinding?.wrappedValue = hasMaskNow
                    context.coordinator.hasClipFeaturesBinding?.wrappedValue = hasFeaturesNow
                    context.coordinator.hasCLIPModelsBinding?.wrappedValue = hasCLIPModelsNow
                    context.coordinator.deformFPSBinding?.wrappedValue = deformFPSNow
                    context.coordinator.renderFPSBinding?.wrappedValue = renderFPSNow
                    context.coordinator.recommendedMaskPercentageBinding?.wrappedValue = recommendedNow
                }
                
                // Set up callback if needed
                if renderer.onEncodingComplete == nil {
                    setupCLIPCallbacks(renderer: renderer, coordinator: context.coordinator)
                }
            }
            
            if captureRequest {
                print("[CLIP-DEBUG] captureRequest=true in updateNSView, setting captureNextFrame")
                context.coordinator.renderer?.captureNextFrame = true
                DispatchQueue.main.async {
                    isEncodingClusters = true
                    captureRequest = false
                }
            }
            
            if searchRequest {
                print("[CLIP-DEBUG] searchRequest detected, dispatching query")
                let coordinator = context.coordinator
                DispatchQueue.main.async {
                    coordinator.processQuery()
                    coordinator.searchRequestBinding?.wrappedValue = false
                }
            }
            
            if saveMaskRequest {
                context.coordinator.renderer?.saveRecommendedMaskPercentage(maskThreshold)
                DispatchQueue.main.async {
                    self.saveMaskRequest = false
                }
            }
            
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
                    if coordinator?.isSelectingModeBinding?.wrappedValue == true {
                        renderer.toggleClusterSelection(clusterID)
                        coordinator?.selectedClusterCountBinding?.wrappedValue = renderer.selectedClusterCount
                    }
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
            
            guard let renderer = MetalKitSceneRenderer(metalKitView) else {
                return metalKitView
            }
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
            
            loadModel(renderer, coordinator: context.coordinator)
            setupCLIPCallbacks(renderer: renderer, coordinator: context.coordinator)
            
            return metalKitView
        }
        
        func updateUIView(_ view: MTKView, context: Context) {
            context.coordinator.renderer?.manualTime = manualTime
            context.coordinator.hasClustersBinding = $hasClusters
            context.coordinator.hasCLIPModelsBinding = $hasCLIPModels
            context.coordinator.isSelectingModeBinding = $isSelectingMode
            context.coordinator.selectedClusterCountBinding = $selectedClusterCount
            context.coordinator.isEncodingClustersBinding = $isEncodingClusters
            context.coordinator.hasClipFeaturesBinding = $hasClipFeatures
            context.coordinator.queryTextBinding = $queryText
            context.coordinator.encodingProgressTextBinding = $encodingProgressText
            context.coordinator.searchRequestBinding = $searchRequest
            context.coordinator.queryStatusTextBinding = $queryStatusText
            context.coordinator.recommendedMaskPercentageBinding = $recommendedMaskPercentage
            
            if let showClusterColors {
                context.coordinator.renderer?.showClusterColors = showClusterColors
            }
            if let showMask {
                context.coordinator.renderer?.showMask = showMask
            }
            if let showDepthVisualization {
                context.coordinator.renderer?.showDepthVisualization = showDepthVisualization
            }
            context.coordinator.renderer?.selectedClusterID = selectedClusterID
            context.coordinator.renderer?.coordinateMode = coordinateMode

            context.coordinator.renderer?.useMaskedCrops = useMaskedCrops
            context.coordinator.renderer?.averageMaskedAndUnmasked = averageMaskedAndUnmasked
            context.coordinator.renderer?.useMaskedDeformation = useMaskedDeformation
            
            if let threshold = context.coordinator.renderer?.getMaskThreshold(forPercentage: maskThreshold) {
                context.coordinator.renderer?.maskThreshold = threshold
            }

            // Pass selection mode to renderer
            let uiMode: UInt32 = isSelectingMode ? 1 : (selectedClusterCount > 0 ? (deleteSelected ? 3 : 2) : 0)
            context.coordinator.renderer?.selectionMode = uiMode
            
            // Update hasClusters state from renderer and handle selection
            if let renderer = context.coordinator.renderer {
                if selectedClusterCount == 0 && !renderer.selectedClusters.isEmpty {
                    renderer.clearSelection()
                }
                let hasClustersNow = renderer.hasClusters
                let hasMaskNow = renderer.hasMask
                let hasFeaturesNow = renderer.clipService.hasFeatures
                let hasCLIPModelsNow = renderer.hasCLIPModels
                let deformFPSNow = renderer.deformFPS
                let renderFPSNow = renderer.renderFPS
                let recommendedNow = renderer.recommendedMaskPercentage
                DispatchQueue.main.async {
                    context.coordinator.hasClustersBinding?.wrappedValue = hasClustersNow
                    context.coordinator.hasMaskBinding?.wrappedValue = hasMaskNow
                    context.coordinator.hasClipFeaturesBinding?.wrappedValue = hasFeaturesNow
                    context.coordinator.hasCLIPModelsBinding?.wrappedValue = hasCLIPModelsNow
                    context.coordinator.deformFPSBinding?.wrappedValue = deformFPSNow
                    context.coordinator.renderFPSBinding?.wrappedValue = renderFPSNow
                    context.coordinator.recommendedMaskPercentageBinding?.wrappedValue = recommendedNow
                }
                
                // Set up callback if needed
                if renderer.onEncodingComplete == nil {
                    setupCLIPCallbacks(renderer: renderer, coordinator: context.coordinator)
                }
            }
            if captureRequest {
                print("[CLIP-DEBUG] captureRequest=true in updateUIView, setting captureNextFrame")
                context.coordinator.renderer?.captureNextFrame = true
                DispatchQueue.main.async {
                    isEncodingClusters = true
                    captureRequest = false
                }
            }
            
            if searchRequest {
                print("[CLIP-DEBUG] searchRequest detected in updateUIView, dispatching query")
                let coordinator = context.coordinator
                DispatchQueue.main.async {
                    coordinator.processQuery()
                    coordinator.searchRequestBinding?.wrappedValue = false
                }
            }
            
            if saveMaskRequest {
                context.coordinator.renderer?.saveRecommendedMaskPercentage(maskThreshold)
                DispatchQueue.main.async {
                    self.saveMaskRequest = false
                }
            }
            
            updateView(context.coordinator)
        }
        
        
#endif // os(iOS)
        
        private func loadModel(_ renderer: MetalKitSceneRenderer?, coordinator: Coordinator) {
            Task { @MainActor in
                do {
                    if let modelIdentifier = modelIdentifier {
                        try await renderer?.load(modelIdentifier)
                        // Update hasClusters after load completes
                        coordinator.hasClustersBinding?.wrappedValue = renderer?.hasClusters ?? false
                        coordinator.hasMaskBinding?.wrappedValue = renderer?.hasMask ?? false
                        coordinator.hasCLIPModelsBinding?.wrappedValue = renderer?.hasCLIPModels ?? false
                    }
                } catch {
                    print("Error loading model: \(error.localizedDescription)")
                }
            }
        }
        
        private func updateView(_ coordinator: Coordinator) {
            guard let renderer = coordinator.renderer else { return }
            Task { @MainActor in
                do {
                    if let modelIdentifier = modelIdentifier {
                        try await renderer.load(modelIdentifier)
                        // Update hasClusters after load completes
                        coordinator.hasClustersBinding?.wrappedValue = renderer.hasClusters
                        coordinator.hasMaskBinding?.wrappedValue = renderer.hasMask
                        coordinator.hasCLIPModelsBinding?.wrappedValue = renderer.hasCLIPModels
                    }
                } catch {
                    print("Error loading model: \(error.localizedDescription)")
                }
            }
        }
    }
}

#endif // os(iOS) || os(macOS)
