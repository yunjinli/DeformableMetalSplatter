import SwiftUI
import RealityKit
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isPickingFile = false
    @State private var useFP16: Bool = true  // Default to FP16 for speed

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(macOS) || os(visionOS)
        mainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            mainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    MetalKitSceneView(modelIdentifier: modelIdentifier)
                        .navigationTitle(modelIdentifier.description)
                }
        }
#endif // os(iOS)
    }

    @ViewBuilder
    var mainView: some View {
        VStack {
            Spacer()

            Text("Deformable MetalSplatter SampleApp")

            Spacer()
            
            // Precision toggle (FP16 vs FP32)
            HStack(spacing: 12) {
                Text("Precision:")
                    .foregroundStyle(.secondary)
                
                Picker("", selection: $useFP16) {
                    Text("FP16").tag(true)
                    Text("FP32").tag(false)
                }
                .pickerStyle(.segmented)
                .frame(width: 120)
                

            }
            .padding(.bottom, 8)

            Button("Read Scene File / Folder") { // Update label
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif
            .fileImporter(isPresented: $isPickingFile,
                          allowedContentTypes: [
                            UTType(filenameExtension: "ply")!,
                            UTType(filenameExtension: "splat")!,
                            UTType.folder // <--- ADD THIS
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    // Security scoping works for folders too
                    _ = url.startAccessingSecurityScopedResource()
                    Task {
                        // Keep access alive longer for folders since we might read large weights
                        try await Task.sleep(for: .seconds(60))
                        url.stopAccessingSecurityScopedResource()
                    }
                    openWindow(value: ModelIdentifier.gaussianSplat(url, useFP16: useFP16))
                case .failure:
                    break
                }
            }

#if os(visionOS)
            Button("Dismiss Immersive Space") {
                Task {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
            .disabled(!immersiveSpaceIsShown)

            Spacer()
#endif // os(visionOS)
        }
    }
}

