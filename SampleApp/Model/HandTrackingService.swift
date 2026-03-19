#if os(iOS)

import AVFoundation
import Vision
import Combine
import UIKit

/// Hand gesture tracking using the front camera and Vision framework.
/// Tracks index finger tip for cursor position, pinch (thumb+index) for selection,
/// and open palm for "explode" removal gesture.
class HandTrackingService: NSObject, ObservableObject {

    // MARK: - Published state

    /// Normalized cursor position on screen (0,0 = top-left, 1,1 = bottom-right).
    @Published var cursorPoint: CGPoint = CGPoint(x: 0.5, y: 0.5)

    /// True while a pinch (thumb + index close together) is detected.
    @Published var isPinching: Bool = false

    /// Fires on double-pinch (two quick pinches). Resets automatically.
    @Published var didDoublePinch: Bool = false

    /// True when an open palm (all fingers extended) is detected.
    @Published var isOpenPalm: Bool = false

    /// Fires once when open palm is first detected. Resets automatically.
    @Published var didOpenPalm: Bool = false

    /// Whether the service is actively running.
    @Published var isRunning: Bool = false

    /// Whether a hand is currently detected.
    @Published var handDetected: Bool = false

    /// Edge rotation: non-zero when cursor is near a screen edge.
    /// x = horizontal rotation speed (negative=left, positive=right).
    /// y = vertical rotation speed (negative=up, positive=down).
    @Published var edgeRotation: CGPoint = .zero

    // MARK: - Configuration

    /// How close to the edge (0…1 normalized) the cursor must be to trigger rotation.
    var edgeZone: CGFloat = 0.1

    /// Maximum rotation speed (radians per frame) when at the very edge.
    var edgeRotationSpeed: CGFloat = 0.015

    /// Distance threshold (normalized) for thumb-to-index to count as a pinch.
    /// Lower = fingers must be closer together. 0.03 requires a very deliberate pinch.
    var pinchThreshold: CGFloat = 0.03

    /// Minimum duration (seconds) a pinch must be held to count as intentional.
    var minPinchDuration: TimeInterval = 0.15

    /// Maximum seconds between two pinch releases to count as a double-pinch.
    var doublePinchWindow: TimeInterval = 0.7

    /// Smoothing factor for cursor (0 = no smoothing, 1 = frozen).
    var smoothingFactor: CGFloat = 0.6

    /// Current interface orientation for coordinate mapping.
    var interfaceOrientation: UIInterfaceOrientation = .portrait

    /// Fraction of fingers that must be extended to count as open palm.
    /// We check index, middle, ring, little (4 fingers). All 4 must be extended.
    var openPalmFingerThreshold: Int = 4

    // MARK: - Private

    private var captureSession: AVCaptureSession?
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "HandTrackingService.processing", qos: .userInteractive)
    private let handPoseRequest = VNDetectHumanHandPoseRequest()

    private var smoothedX: CGFloat = 0.5
    private var smoothedY: CGFloat = 0.5

    // Pinch state machine
    private var wasPinching = false
    private var pinchStartTime: TimeInterval = 0
    private var lastPinchEndTime: TimeInterval = 0
    private var pinchCount = 0

    // Open palm state machine (double open-palm required)
    private var wasOpenPalm = false
    private var lastOpenPalmEndTime: TimeInterval = 0
    private var openPalmCount = 0

    /// Maximum seconds between two open-palm gestures to count as a double.
    var doubleOpenPalmWindow: TimeInterval = 0.8

    // MARK: - Lifecycle

    override init() {
        super.init()
        handPoseRequest.maximumHandCount = 1
    }

    func start() {
        guard !isRunning else { return }

        let session = AVCaptureSession()
        session.sessionPreset = .low  // Low res is fine for hand pose

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("[HandTracking] Failed to access front camera")
            return
        }

        guard session.canAddInput(input) else { return }
        session.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else { return }
        session.addOutput(videoOutput)


        captureSession = session
        smoothedX = 0.5
        smoothedY = 0.5
        wasPinching = false
        wasOpenPalm = false
        pinchCount = 0

        processingQueue.async {
            session.startRunning()
        }
        isRunning = true
        print("[HandTracking] Started hand tracking session")
    }

    func stop() {
        processingQueue.async { [weak self] in
            self?.captureSession?.stopRunning()
        }
        captureSession = nil
        DispatchQueue.main.async {
            self.isRunning = false
            self.handDetected = false
            self.isPinching = false
            self.isOpenPalm = false
        }
        print("[HandTracking] Stopped hand tracking session")
    }

    /// Re-center the cursor to screen center.
    func recenter() {
        smoothedX = 0.5
        smoothedY = 0.5
    }

}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HandTrackingService: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Detect current UI orientation
        var uiOrientation: UIInterfaceOrientation = .portrait
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            uiOrientation = scene.interfaceOrientation
        }
        interfaceOrientation = uiOrientation

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Map UI orientation to CGImagePropertyOrientation for the front (mirrored) camera.
        // This tells Vision how the image is oriented so it returns coordinates
        // in the correct screen-relative frame.
        let cgOrientation: CGImagePropertyOrientation
        switch uiOrientation {
        case .landscapeLeft:
            cgOrientation = .downMirrored
        case .landscapeRight:
            cgOrientation = .upMirrored
        case .portraitUpsideDown:
            cgOrientation = .leftMirrored
        default:
            cgOrientation = .rightMirrored
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: cgOrientation, options: [:])
        do {
            try handler.perform([handPoseRequest])
        } catch {
            return
        }

        guard let observation = handPoseRequest.results?.first else {
            DispatchQueue.main.async {
                self.handDetected = false
            }
            return
        }

        processHandObservation(observation)
    }

    private func processHandObservation(_ observation: VNHumanHandPoseObservation) {
        guard let indexTip = try? observation.recognizedPoint(.indexTip),
              let thumbTip = try? observation.recognizedPoint(.thumbTip),
              indexTip.confidence > 0.3, thumbTip.confidence > 0.3 else {
            DispatchQueue.main.async { self.handDetected = false }
            return
        }

        // --- Cursor position from index finger tip ---
        // With the correct CGImagePropertyOrientation passed to VNImageRequestHandler,
        // Vision returns normalized coords in screen-relative frame:
        // origin bottom-left, y-up. We just flip y for screen top-left origin.
        let rawX = indexTip.location.x
        let rawY = 1.0 - indexTip.location.y

        let newX = smoothedX * smoothingFactor + rawX * (1.0 - smoothingFactor)
        let newY = smoothedY * smoothingFactor + rawY * (1.0 - smoothingFactor)
        smoothedX = max(0, min(1, newX))
        smoothedY = max(0, min(1, newY))

        // --- Pinch detection (thumb tip to index tip distance) ---
        let dx = thumbTip.location.x - indexTip.location.x
        let dy = thumbTip.location.y - indexTip.location.y
        let distance = sqrt(dx * dx + dy * dy)
        let currentlyPinching = distance < pinchThreshold

        // --- Open palm detection (all 4 fingers extended) ---
        let extendedCount = countExtendedFingers(observation)
        let currentlyOpenPalm = extendedCount >= openPalmFingerThreshold && !currentlyPinching

        let now = CACurrentMediaTime()

        // Edge rotation: compute before dispatching
        let edgeZone = self.edgeZone
        let edgeSpeed = self.edgeRotationSpeed
        var edgeX: CGFloat = 0
        var edgeY: CGFloat = 0
        if smoothedX < edgeZone {
            edgeX = -edgeSpeed * (1.0 - smoothedX / edgeZone)
        } else if smoothedX > (1.0 - edgeZone) {
            edgeX = edgeSpeed * (1.0 - (1.0 - smoothedX) / edgeZone)
        }
        if smoothedY < edgeZone {
            edgeY = -edgeSpeed * (1.0 - smoothedY / edgeZone)
        } else if smoothedY > (1.0 - edgeZone) {
            edgeY = edgeSpeed * (1.0 - (1.0 - smoothedY) / edgeZone)
        }

        DispatchQueue.main.async {
            self.handDetected = true
            self.cursorPoint = CGPoint(x: self.smoothedX, y: self.smoothedY)
            self.edgeRotation = CGPoint(x: edgeX, y: edgeY)
            self.isPinching = currentlyPinching

            // Track pinch start time
            if currentlyPinching && !self.wasPinching {
                self.pinchStartTime = now
            }

            // Double-pinch: detect on pinch release, only if held long enough
            if self.wasPinching && !currentlyPinching {
                let pinchDuration = now - self.pinchStartTime
                if pinchDuration >= self.minPinchDuration {
                    // This was an intentional pinch
                    if now - self.lastPinchEndTime < self.doublePinchWindow {
                        self.pinchCount += 1
                    } else {
                        self.pinchCount = 1
                    }
                    self.lastPinchEndTime = now

                    if self.pinchCount >= 2 {
                        self.didDoublePinch = true
                        self.pinchCount = 0
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                            self.didDoublePinch = false
                        }
                    }
                }
                // else: too brief, ignore as noise
            }
            self.wasPinching = currentlyPinching

            // Open palm: require double (open-close-open) to trigger
            self.isOpenPalm = currentlyOpenPalm
            if self.wasOpenPalm && !currentlyOpenPalm {
                // Palm just closed — count it
                if now - self.lastOpenPalmEndTime < self.doubleOpenPalmWindow {
                    self.openPalmCount += 1
                } else {
                    self.openPalmCount = 1
                }
                self.lastOpenPalmEndTime = now

                if self.openPalmCount >= 2 {
                    self.didOpenPalm = true
                    self.openPalmCount = 0
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        self.didOpenPalm = false
                    }
                }
            }
            self.wasOpenPalm = currentlyOpenPalm
        }
    }

    /// Count how many of the 4 fingers (index, middle, ring, little) are extended.
    /// A finger is "extended" if its tip is significantly above (in Vision y-coords) its PIP joint.
    private func countExtendedFingers(_ observation: VNHumanHandPoseObservation) -> Int {
        let fingers: [(tip: VNHumanHandPoseObservation.JointName, pip: VNHumanHandPoseObservation.JointName)] = [
            (.indexTip, .indexPIP),
            (.middleTip, .middlePIP),
            (.ringTip, .ringPIP),
            (.littleTip, .littlePIP),
        ]

        var count = 0
        for (tip, pip) in fingers {
            guard let tipPt = try? observation.recognizedPoint(tip),
                  let pipPt = try? observation.recognizedPoint(pip),
                  tipPt.confidence > 0.2, pipPt.confidence > 0.2 else { continue }
            // In Vision coords, y increases upward. Extended finger: tip.y > pip.y
            if tipPt.location.y > pipPt.location.y + 0.02 {
                count += 1
            }
        }
        return count
    }
}

#endif // os(iOS)
