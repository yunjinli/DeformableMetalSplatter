#if os(iOS)

import ARKit
import Combine
import simd

/// Provides head-pose-based pointer and blink detection using ARKit face tracking (TrueDepth camera).
/// The pointer position is derived from head yaw/pitch relative to the initial pose when tracking started.
/// Blink (both eyes closed then reopened) triggers a selection event.
class EyeTrackingService: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: - Published state

    /// Normalized pointer position on screen (0,0 = top-left, 1,1 = bottom-right).
    @Published var gazePoint: CGPoint = CGPoint(x: 0.5, y: 0.5)

    /// True while both eyes are detected as blinking.
    @Published var isBlinking: Bool = false

    /// True if a blink-select event just fired (resets automatically).
    @Published var didBlink: Bool = false

    /// Whether the service is actively running.
    @Published var isRunning: Bool = false

    /// Whether ARFaceTracking is supported on this device.
    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    // MARK: - Configuration

    /// Blend-shape threshold above which we consider the eye closed.
    var blinkThreshold: Float = 0.6

    /// Minimum seconds between blink-select triggers to avoid rapid-fire.
    var blinkCooldown: TimeInterval = 0.8

    /// Smoothing factor for pointer (0 = no smoothing, 1 = frozen).
    var smoothingFactor: Float = 0.6

    /// How many degrees of head turn maps to full screen width/height.
    /// Smaller = more sensitive, larger = need bigger head movements.
    var degreesForFullScreen: Float = 30.0

    /// Current interface orientation — set by the caller so we can remap axes for landscape.
    var interfaceOrientation: UIInterfaceOrientation = .portrait

    // MARK: - Private

    private let session = ARSession()
    private var lastBlinkTime: TimeInterval = 0
    private var wasBlinking = false

    // Reference pose captured on start (center of screen)
    private var referenceYaw: Float?
    private var referencePitch: Float?

    // Smoothed normalized values
    private var smoothedX: Float = 0.5
    private var smoothedY: Float = 0.5

    // Screen size (unused in head-tracking mode but kept for potential calibration)
    var screenSize: CGSize = CGSize(width: 390, height: 844)

    // MARK: - Lifecycle

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard EyeTrackingService.isSupported else {
            print("[HeadTracking] ARFaceTracking not supported on this device")
            return
        }
        // Reset reference so the first frame captures current head pose as center
        referenceYaw = nil
        referencePitch = nil
        smoothedX = 0.5
        smoothedY = 0.5

        let config = ARFaceTrackingConfiguration()
        if ARFaceTrackingConfiguration.supportsWorldTracking {
            config.isWorldTrackingEnabled = false
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        print("[HeadTracking] Started face tracking session")
    }

    func stop() {
        session.pause()
        isRunning = false
        isBlinking = false
        didBlink = false
        wasBlinking = false
        referenceYaw = nil
        referencePitch = nil
        print("[HeadTracking] Stopped face tracking session")
    }

    /// Re-center the pointer to the current head position.
    func recenter() {
        referenceYaw = nil
        referencePitch = nil
        smoothedX = 0.5
        smoothedY = 0.5
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.compactMap({ $0 as? ARFaceAnchor }).first else { return }

        // Update orientation each frame so rotation mid-session works
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            interfaceOrientation = scene.interfaceOrientation
        }

        // --- Head pose → pointer ---
        // Extract yaw and pitch from the face anchor's world transform.
        let transform = faceAnchor.transform
        let yaw = atan2(transform.columns.0.z, transform.columns.2.z)   // left/right turn
        let pitch = asin(-transform.columns.1.z)                         // up/down nod

        // Capture initial pose as the reference (screen center)
        if referenceYaw == nil {
            referenceYaw = yaw
            referencePitch = pitch
        }

        let deltaYaw = yaw - (referenceYaw ?? yaw)
        let deltaPitch = pitch - (referencePitch ?? pitch)

        // Remap head yaw/pitch to screen horizontal/vertical based on device orientation.
        // ARKit face tracking always reports in the camera's native (portrait) frame.
        // In landscape the screen axes are rotated relative to the head axes.
        let screenH: Float  // positive = pointer moves right on screen
        let screenV: Float  // positive = pointer moves down on screen
        switch interfaceOrientation {
        case .landscapeLeft:
            // Home button on left: head turn right → cursor right, head nod down → cursor down
            screenH = -deltaYaw
            screenV = -deltaPitch
        case .landscapeRight:
            // Home button on right: head turn right → cursor right, head nod down → cursor down
            screenH = -deltaYaw
            screenV = -deltaPitch
        case .portraitUpsideDown:
            screenH = -deltaYaw
            screenV = deltaPitch
        default:
            // Portrait
            screenH = -deltaYaw
            screenV = -deltaPitch
        }

        let range = degreesForFullScreen * .pi / 180.0
        let rawX = 0.5 + (screenH / range) * 0.5
        let rawY = 0.5 + (screenV / range) * 0.5

        // Exponential smoothing
        smoothedX = smoothedX * smoothingFactor + rawX * (1.0 - smoothingFactor)
        smoothedY = smoothedY * smoothingFactor + rawY * (1.0 - smoothingFactor)

        let clampedX = max(0.0, min(1.0, smoothedX))
        let clampedY = max(0.0, min(1.0, smoothedY))

        // --- Blink detection ---
        let blendShapes = faceAnchor.blendShapes
        let leftBlink = (blendShapes[.eyeBlinkLeft]?.floatValue) ?? 0
        let rightBlink = (blendShapes[.eyeBlinkRight]?.floatValue) ?? 0
        let bothClosed = leftBlink > blinkThreshold && rightBlink > blinkThreshold
        let now = CACurrentMediaTime()

        DispatchQueue.main.async {
            self.gazePoint = CGPoint(x: CGFloat(clampedX), y: CGFloat(clampedY))
            self.isBlinking = bothClosed

            // Trigger on blink release (eyes reopen after being closed)
            if self.wasBlinking && !bothClosed {
                if now - self.lastBlinkTime > self.blinkCooldown {
                    self.lastBlinkTime = now
                    self.didBlink = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        self.didBlink = false
                    }
                }
            }
            self.wasBlinking = bothClosed
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[HeadTracking] Session failed: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }
}

#endif // os(iOS)
