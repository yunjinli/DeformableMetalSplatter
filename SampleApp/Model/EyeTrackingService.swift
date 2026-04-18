#if os(iOS) || os(macOS)

import Foundation
import Combine
import simd

#if os(iOS)
import ARKit
#elseif os(macOS)
import AVFoundation
import Vision
import CoreMedia
#endif

/// Publishes smoothed head yaw/pitch deltas (in radians) relative to the pose captured on start.
/// On iOS this is sourced from ARKit face tracking (TrueDepth camera). On macOS it uses the
/// FaceTime/front camera via AVCaptureSession and Vision's `VNDetectFaceRectanglesRequest`,
/// which exposes roll/yaw/pitch directly. Consumers feed these deltas into a scene rotation
/// controller to orbit the view in the same spirit as gyroscope-based "fake depth" mode.
class EyeTrackingService: NSObject, ObservableObject {

    // MARK: - Published state

    /// Smoothed head yaw delta in radians, relative to the reference pose
    /// (positive = head turns left from the viewer's perspective).
    @Published var headYaw: Float = 0

    /// Smoothed head pitch delta in radians, relative to the reference pose
    /// (positive = head tilts up).
    @Published var headPitch: Float = 0

    /// Whether the service is actively running.
    @Published var isRunning: Bool = false

    /// Whether head tracking is supported on this device.
    static var isSupported: Bool {
        #if os(iOS)
        return ARFaceTrackingConfiguration.isSupported
        #elseif os(macOS)
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) != nil
            || AVCaptureDevice.default(for: .video) != nil
        #endif
    }

    // MARK: - Configuration

    /// Smoothing factor (0 = no smoothing, 1 = frozen).
    var smoothingFactor: Float = 0.6

    // MARK: - Private (shared)

    // Reference pose captured on start
    private var referenceYaw: Float?
    private var referencePitch: Float?

    // Smoothed radian deltas
    private var smoothedYaw: Float = 0
    private var smoothedPitch: Float = 0

    // MARK: - Private (iOS)

    #if os(iOS)
    private let session = ARSession()
    #endif

    // MARK: - Private (macOS)

    #if os(macOS)
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "HeadTrackingService.sampleQueue",
                                            qos: .userInitiated)
    private var isConfigured = false
    #endif

    // MARK: - Lifecycle

    override init() {
        super.init()
        #if os(iOS)
        session.delegate = self
        #endif
    }

    func start() {
        guard EyeTrackingService.isSupported else {
            print("[HeadTracking] Not supported on this device")
            return
        }
        resetState()

        #if os(iOS)
        let config = ARFaceTrackingConfiguration()
        if ARFaceTrackingConfiguration.supportsWorldTracking {
            config.isWorldTrackingEnabled = false
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        print("[HeadTracking] Started ARKit face tracking session")
        #elseif os(macOS)
        startMacCamera()
        #endif
    }

    func stop() {
        #if os(iOS)
        session.pause()
        #elseif os(macOS)
        if captureSession.isRunning {
            sampleQueue.async { [captureSession] in
                captureSession.stopRunning()
            }
        }
        #endif
        isRunning = false
        referenceYaw = nil
        referencePitch = nil
        DispatchQueue.main.async {
            self.headYaw = 0
            self.headPitch = 0
        }
        print("[HeadTracking] Stopped session")
    }

    /// Re-anchor the reference pose to the current head position.
    func recenter() {
        resetState()
    }

    private func resetState() {
        referenceYaw = nil
        referencePitch = nil
        smoothedYaw = 0
        smoothedPitch = 0
    }

    /// Process a raw yaw/pitch reading (radians) from whichever backend is active.
    fileprivate func processPose(yaw: Float, pitch: Float) {
        if referenceYaw == nil {
            referenceYaw = yaw
            referencePitch = pitch
        }

        let deltaYaw = yaw - (referenceYaw ?? yaw)
        let deltaPitch = pitch - (referencePitch ?? pitch)

        smoothedYaw = smoothedYaw * smoothingFactor + deltaYaw * (1.0 - smoothingFactor)
        smoothedPitch = smoothedPitch * smoothingFactor + deltaPitch * (1.0 - smoothingFactor)

        let outYaw = smoothedYaw
        let outPitch = smoothedPitch
        DispatchQueue.main.async {
            self.headYaw = outYaw
            self.headPitch = outPitch
        }
    }
}

// MARK: - iOS ARKit backend

#if os(iOS)
extension EyeTrackingService: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.compactMap({ $0 as? ARFaceAnchor }).first else { return }

        let transform = faceAnchor.transform
        let yaw = atan2(transform.columns.0.z, transform.columns.2.z)
        let pitch = asin(-transform.columns.1.z)
        processPose(yaw: yaw, pitch: pitch)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[HeadTracking] Session failed: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }
}
#endif

// MARK: - macOS AVFoundation + Vision backend

#if os(macOS)
extension EyeTrackingService: AVCaptureVideoDataOutputSampleBufferDelegate {

    fileprivate func startMacCamera() {
        let proceed: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.sampleQueue.async {
                self.configureSessionIfNeeded()
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
                DispatchQueue.main.async {
                    self.isRunning = self.captureSession.isRunning
                }
                if self.captureSession.isRunning {
                    print("[HeadTracking] Started Vision face tracking on macOS")
                }
            }
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            proceed()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    proceed()
                } else {
                    print("[HeadTracking] Camera access denied")
                }
            }
        case .denied, .restricted:
            print("[HeadTracking] Camera access denied/restricted; enable it in System Settings → Privacy & Security → Camera")
        @unknown default:
            break
        }
    }

    private func configureSessionIfNeeded() {
        guard !isConfigured else { return }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium

        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)

        guard let camera = device,
              let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            print("[HeadTracking] Failed to configure front camera input")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                        Int(kCVPixelFormatType_32BGRA)]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Keep the raw (non-mirrored) sensor feed so yaw signs are consistent with a
        // camera-forward convention; we invert yaw below to match the iOS "head-left = +yaw"
        // semantics expected by the scene rotation consumer.
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }

        captureSession.commitConfiguration()
        isConfigured = true
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNDetectFaceRectanglesRequest { [weak self] request, _ in
            guard let self = self,
                  let observations = request.results as? [VNFaceObservation] else { return }

            // Pick the largest face (closest to the camera) for stability.
            let face = observations.max { lhs, rhs in
                lhs.boundingBox.width * lhs.boundingBox.height
                    < rhs.boundingBox.width * rhs.boundingBox.height
            }
            guard let face = face,
                  let yawN = face.yaw,
                  let pitchN = face.pitch else { return }

            // VNFaceObservation yaw: positive = head turns to the camera's right
            // (equivalent to the viewer's left when looking at a non-mirrored feed).
            // Flip so positive yaw = viewer's right, matching the iOS convention we
            // consume in MetalKitSceneView (which negates again on apply).
            let yaw = -yawN.floatValue
            let pitch = pitchN.floatValue
            self.processPose(yaw: yaw, pitch: pitch)
        }
        request.revision = VNDetectFaceRectanglesRequestRevision3

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        try? handler.perform([request])
    }
}
#endif

#endif // os(iOS) || os(macOS)
