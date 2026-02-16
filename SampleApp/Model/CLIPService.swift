#if os(iOS) || os(macOS)

import CoreML
import CoreImage
import Metal
import os

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Manages MobileCLIP2-S0 CoreML models for on-device CLIP inference.
/// Handles both image encoding (per-cluster crops) and text encoding (query).
public class CLIPService {
    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "CLIPService",
        category: "CLIPService"
    )

    // MARK: - Feature dimension
    static let featureDim = 512

    // MARK: - Models
    private var imageEncoder: MLModel?
    private var textEncoder: MLModel?

    // MARK: - Model I/O names
    private var imageInputName = "image"
    private var imageOutputName = "features"
    private var textInputName = "input_ids"
    private var textOutputName = "features"

    // MARK: - Tokenizer
    private var vocab: [String: Int] = [:]
    private var merges: [(String, String)] = []
    private let sotToken = 49406
    private let eotToken = 49407
    private let contextLength = 77

    // MARK: - State
    /// Per-cluster L2-normalized feature vectors (N × 512), stored as flat Float arrays.
    /// Protected by `featureStateLock` because encoding happens off-main-thread while querying runs on main.
    private var clusterFeaturesStorage: [[Float]] = []
    /// Cluster IDs corresponding to each feature row.
    /// Protected by `featureStateLock`.
    private var clusterIDsStorage: [Int32] = []
    private let featureStateLock = NSLock()

    private(set) var clusterFeatures: [[Float]] {
        get {
            featureStateLock.lock()
            defer { featureStateLock.unlock() }
            return clusterFeaturesStorage
        }
        set {
            featureStateLock.lock()
            clusterFeaturesStorage = newValue
            featureStateLock.unlock()
        }
    }

    private(set) var clusterIDs: [Int32] {
        get {
            featureStateLock.lock()
            defer { featureStateLock.unlock() }
            return clusterIDsStorage
        }
        set {
            featureStateLock.lock()
            clusterIDsStorage = newValue
            featureStateLock.unlock()
        }
    }

    var encodedClusterCount: Int {
        featureStateLock.lock()
        defer { featureStateLock.unlock() }
        return min(clusterFeaturesStorage.count, clusterIDsStorage.count)
    }

    var hasFeatures: Bool { encodedClusterCount > 0 }
    var hasImageEncoder: Bool { imageEncoder != nil }
    var hasTextEncoder: Bool { textEncoder != nil }

    // MARK: - Progress tracking
    /// (encoded, total) — updated during encoding
    private(set) var encodingProgress: (encoded: Int, total: Int) = (0, 0)
    /// Called on main thread with a descriptive status string during encoding
    var onStatusUpdate: ((String) -> Void)?

    // MARK: - Init

    init() {
        loadModels()
        loadTokenizer()
        
        // Prominent startup diagnostics
        let imgStatus = imageEncoder != nil ? "✓ LOADED" : "✗ MISSING"
        let txtStatus = textEncoder != nil ? "✓ LOADED" : "✗ MISSING"
        print("╔══════════════════════════════════════════════════╗")
        print("║  CLIPService Startup                            ║")
        print("║  Image Encoder: \(imgStatus)                    ║")
        print("║  Text  Encoder: \(txtStatus)                    ║")
        print("║  Vocab: \(vocab.count) tokens, Merges: \(merges.count)        ║")
        print("╚══════════════════════════════════════════════════╝")
        if imageEncoder == nil {
            print("[CLIP] ⚠️ MobileCLIPImageEncoder.mlmodelc NOT found in bundle — encoding will produce 0 features!")
            print("[CLIP] Make sure the .mlpackage is added to Sources build phase and do a CLEAN BUILD.")
        }
        if textEncoder == nil {
            print("[CLIP] ⚠️ MobileCLIP2TextEncoder.mlmodelc NOT found in bundle — text queries will fail!")
        }
    }

    // MARK: - Model loading

    private func loadModels() {
        // Look for models in the app bundle
        if let imageURL = Bundle.main.url(forResource: "MobileCLIPImageEncoder", withExtension: "mlmodelc") {
            loadImageEncoder(from: imageURL)
        } else {
            Self.log.warning("MobileCLIPImageEncoder.mlmodelc not found in bundle")
        }

        if let textURL = Bundle.main.url(forResource: "MobileCLIP2TextEncoder", withExtension: "mlmodelc") {
            loadTextEncoder(from: textURL)
        } else {
            Self.log.warning("MobileCLIP2TextEncoder.mlmodelc not found in bundle")
        }
    }

    private func loadImageEncoder(from url: URL) {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            imageEncoder = try MLModel(contentsOf: url, configuration: config)

            // Warmup: run a dummy inference to ensure consistent timing/behavior
            warmupEncoder(isImage: true)

            // Discover I/O names
            if let desc = imageEncoder?.modelDescription {
                if let firstName = desc.inputDescriptionsByName.keys.first {
                    imageInputName = firstName
                }
                if let firstName = desc.outputDescriptionsByName.keys.first {
                    imageOutputName = firstName
                }
            }
            Self.log.info("Image encoder loaded: in='\(self.imageInputName)' out='\(self.imageOutputName)'")
        } catch {
            Self.log.error("Failed to load image encoder: \(error.localizedDescription)")
        }
    }

    private func loadTextEncoder(from url: URL) {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            textEncoder = try MLModel(contentsOf: url, configuration: config)

            // Warmup: run a dummy inference to ensure consistent timing/behavior
            warmupEncoder(isImage: false)

            if let desc = textEncoder?.modelDescription {
                if let firstName = desc.inputDescriptionsByName.keys.first {
                    textInputName = firstName
                }
                if let firstName = desc.outputDescriptionsByName.keys.first {
                    textOutputName = firstName
                }
            }
            Self.log.info("Text encoder loaded: in='\(self.textInputName)' out='\(self.textOutputName)'")
        } catch {
            Self.log.error("Failed to load text encoder: \(error.localizedDescription)")
        }
    }

    /// Warmup the encoder with a dummy input to ensure consistent first-run behavior.
    private func warmupEncoder(isImage: Bool) {
        if isImage {
            guard let model = imageEncoder else { return }
            // Create a dummy 256x256 grayscale image
            guard let dummyImage = createDummyImage(size: 256),
                  let pixelBuffer = cgImageToPixelBuffer(dummyImage, width: 256, height: 256) else {
                Self.log.warning("Failed to create dummy image for warmup")
                return
            }
            do {
                let featureValue = try MLFeatureValue(pixelBuffer: pixelBuffer)
                let input = try MLDictionaryFeatureProvider(dictionary: [imageInputName: featureValue])
                _ = try model.prediction(from: input)
                Self.log.info("Image encoder warmup complete")
            } catch {
                Self.log.warning("Image encoder warmup failed: \(error.localizedDescription)")
            }
        } else {
            guard let model = textEncoder else { return }
            // Create dummy tokens
            let dummyTokens = [Int32](repeating: 0, count: contextLength)
            do {
                let shape: [NSNumber] = [1, NSNumber(value: contextLength)]
                let tokenArray = try MLMultiArray(shape: shape, dataType: .int32)
                for i in 0..<contextLength {
                    tokenArray[i] = NSNumber(value: dummyTokens[i])
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [textInputName: MLFeatureValue(multiArray: tokenArray)])
                _ = try model.prediction(from: input)
                Self.log.info("Text encoder warmup complete")
            } catch {
                Self.log.warning("Text encoder warmup failed: \(error.localizedDescription)")
            }
        }
    }

    /// Create a small dummy RGB CGImage for warmup (model expects 3-channel input)
    private func createDummyImage(size: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: size * size * 3)
        // Fill with zeros (black RGB = 0,0,0)
        guard let context = CGContext(data: &pixelData,
                                       width: size, height: size,
                                       bitsPerComponent: 8, bytesPerRow: size * 3,
                                       space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
        }
        return context.makeImage()
    }

    // MARK: - Tokenizer loading

    private func loadTokenizer() {
        guard let vocabURL = Bundle.main.url(forResource: "clip-vocab", withExtension: "json"),
              let mergesURL = Bundle.main.url(forResource: "clip-merges", withExtension: "txt") else {
            Self.log.warning("Tokenizer files not found in bundle")
            return
        }

        do {
            let vocabData = try Data(contentsOf: vocabURL)
            if let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
                vocab = vocabDict
            }
        } catch {
            Self.log.error("Failed to load vocab: \(error.localizedDescription)")
        }

        do {
            let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
            let lines = mergesText.components(separatedBy: "\n")
            for line in lines {
                if line.hasPrefix("#") || line.isEmpty { continue }
                let parts = line.components(separatedBy: " ")
                if parts.count == 2 {
                    merges.append((parts[0], parts[1]))
                }
            }
        } catch {
            Self.log.error("Failed to load merges: \(error.localizedDescription)")
        }

        Self.log.info("Tokenizer loaded: \(self.vocab.count) vocab, \(self.merges.count) merges")
    }

    // MARK: - BPE Tokenization

    /// Tokenize a text prompt into a (1, 77) int32 array suitable for the CoreML model.
    func tokenize(_ text: String) -> [Int32] {
        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleaned.components(separatedBy: .whitespaces).filter { !$0.isEmpty }

        var allTokens: [Int] = [sotToken]

        for word in words {
            // CLIP tokenizer adds </w> to the end of each word
            let wordWithSuffix = word + "</w>"
            let wordTokens = bpe(wordWithSuffix)
            allTokens.append(contentsOf: wordTokens)
        }

        allTokens.append(eotToken)

        // Pad/truncate to context length
        var result = [Int32](repeating: 0, count: contextLength)
        for i in 0..<min(allTokens.count, contextLength) {
            result[i] = Int32(allTokens[i])
        }
        return result
    }

    /// Byte-pair-encoding for a single word (already has </w> suffix)
    private func bpe(_ token: String) -> [Int] {
        // Split into characters
        var word = token.map { String($0) }
        if word.isEmpty { return [] }

        // Build pair ranking dictionary for O(1) lookup
        var mergeRank: [String: Int] = [:]
        for (i, pair) in merges.enumerated() {
            mergeRank["\(pair.0) \(pair.1)"] = i
        }

        while word.count > 1 {
            // Find best (lowest rank) pair
            var bestPair: (Int, String, String)? = nil
            var bestRank = Int.max

            for i in 0..<(word.count - 1) {
                let key = "\(word[i]) \(word[i+1])"
                if let rank = mergeRank[key], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, word[i], word[i+1])
                }
            }

            guard let pair = bestPair else { break }

            // Merge all occurrences of this pair
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.1 && word[i+1] == pair.2 {
                    newWord.append(pair.1 + pair.2)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        // Look up token IDs
        return word.compactMap { vocab[$0] }
    }

    // MARK: - Image encoding

    /// Encode a CGImage crop into a 512-d L2-normalized feature vector.
    func encodeImage(_ image: CGImage) -> [Float]? {
        guard let model = imageEncoder else {
            Self.log.error("Image encoder not loaded")
            return nil
        }

        // The CoreML model expects a 256×256 RGB image (imageType input)
        let targetSize = 256

        // Resize the image to 256×256
        guard let resized = resizeCGImage(image, to: targetSize) else {
            Self.log.error("Failed to resize image")
            return nil
        }

        // Create CVPixelBuffer from CGImage
        guard let pixelBuffer = cgImageToPixelBuffer(resized, width: targetSize, height: targetSize) else {
            Self.log.error("Failed to create pixel buffer")
            return nil
        }

        do {
            let featureValue = try MLFeatureValue(pixelBuffer: pixelBuffer)
            let input = try MLDictionaryFeatureProvider(dictionary: [imageInputName: featureValue])
            let output = try model.prediction(from: input)

            guard let featuresArray = output.featureValue(for: imageOutputName)?.multiArrayValue else {
                Self.log.error("No output features from image encoder")
                return nil
            }

            // Extract and L2-normalize
            var features = [Float](repeating: 0, count: Self.featureDim)
            let ptr = featuresArray.dataPointer.bindMemory(to: Float.self, capacity: Self.featureDim)
            for i in 0..<Self.featureDim {
                features[i] = ptr[i]
            }
            l2Normalize(&features)
            return features
        } catch {
            Self.log.error("Image encoding failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Text encoding

    /// Encode a text prompt into a 512-d L2-normalized feature vector.
    func encodeText(_ text: String) -> [Float]? {
        guard let model = textEncoder else {
            Self.log.error("Text encoder not loaded")
            return nil
        }

        let tokens = tokenize(text)

        do {
            // Create (1, 77) MLMultiArray of Int32
            let shape: [NSNumber] = [1, NSNumber(value: contextLength)]
            let tokenArray = try MLMultiArray(shape: shape, dataType: .int32)
            for i in 0..<contextLength {
                tokenArray[i] = NSNumber(value: tokens[i])
            }

            let input = try MLDictionaryFeatureProvider(dictionary: [textInputName: MLFeatureValue(multiArray: tokenArray)])
            let output = try model.prediction(from: input)

            guard let featuresArray = output.featureValue(for: textOutputName)?.multiArrayValue else {
                Self.log.error("No output features from text encoder")
                return nil
            }

            var features = [Float](repeating: 0, count: Self.featureDim)
            let ptr = featuresArray.dataPointer.bindMemory(to: Float.self, capacity: Self.featureDim)
            for i in 0..<Self.featureDim {
                features[i] = ptr[i]
            }
            l2Normalize(&features)
            return features
        } catch {
            Self.log.error("Text encoding failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Cluster Encoding Pipeline

    /// Atomically snapshot the currently encoded cluster IDs + features.
    func featuresSnapshot() -> (clusterIDs: [Int32], clusterFeatures: [[Float]]) {
        featureStateLock.lock()
        defer { featureStateLock.unlock() }
        return (clusterIDsStorage, clusterFeaturesStorage)
    }

    /// Atomically replace encoded cluster IDs + features.
    func replaceFeatures(clusterIDs: [Int32], clusterFeatures: [[Float]]) {
        featureStateLock.lock()
        clusterIDsStorage = clusterIDs
        clusterFeaturesStorage = clusterFeatures
        featureStateLock.unlock()
    }

    /// Given RGB and cluster-ID textures from the renderer, extract per-cluster bounding-box crops
    /// and encode each one with the image encoder. Stores results in `clusterFeatures` / `clusterIDs`.
    /// Encode cluster crops. If `averageMaskedAndUnmasked` is true, runs both masked and unmasked
    /// modes and averages the resulting features for improved robustness.
    func encodeClusterCrops(rgb: MTLTexture, clusters: MTLTexture, useMaskedCrops: Bool = false, averageMaskedAndUnmasked: Bool = false) {
        let width = rgb.width
        let height = rgb.height
        print("[CLIP-DEBUG] encodeClusterCrops called: rgb=\(width)x\(height), clusters=\(clusters.width)x\(clusters.height)")
        print("[CLIP-DEBUG] imageEncoder loaded: \(imageEncoder != nil), averageMaskedAndUnmasked=\(averageMaskedAndUnmasked)")

        // Status: reading textures
        DispatchQueue.main.async { [weak self] in
            self?.onStatusUpdate?("Reading textures…")
        }

        // Read RGB texture (BGRA8)
        let bytesPerPixel = 4
        let rowBytes = width * bytesPerPixel
        var rgbData = [UInt8](repeating: 0, count: height * rowBytes)
        rgb.getBytes(&rgbData, bytesPerRow: rowBytes,
                     from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        print("[CLIP-DEBUG] RGB data read: \(rgbData.count) bytes, first 4 bytes: [\(rgbData[0]), \(rgbData[1]), \(rgbData[2]), \(rgbData[3])]")

        // Read cluster-ID texture (R32Sint)
        let clusterRowBytes = width * 4
        var clusterData = [Int32](repeating: -1, count: width * height)
        clusterData.withUnsafeMutableBufferPointer { ptr in
            clusters.getBytes(ptr.baseAddress!, bytesPerRow: clusterRowBytes,
                              from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        
        // Debug: check how many non-negative cluster IDs exist
        let nonNegCount = clusterData.filter { $0 >= 0 }.count
        let uniqueIDs = Set(clusterData.filter { $0 >= 0 })
        print("[CLIP-DEBUG] Cluster data read: \(clusterData.count) pixels, \(nonNegCount) with valid cluster ID, \(uniqueIDs.count) unique clusters")
        if uniqueIDs.count > 0 {
            print("[CLIP-DEBUG] Unique cluster IDs: \(uniqueIDs.sorted())")
        }

        // Find unique cluster IDs, their bounding boxes, and pixel counts
        var bboxes: [Int32: (minX: Int, minY: Int, maxX: Int, maxY: Int)] = [:]
        var pixelCounts: [Int32: Int] = [:]

        for y in 0..<height {
            for x in 0..<width {
                let cid = clusterData[y * width + x]
                if cid < 0 { continue }
                pixelCounts[cid, default: 0] += 1
                if var bb = bboxes[cid] {
                    bb.minX = min(bb.minX, x)
                    bb.minY = min(bb.minY, y)
                    bb.maxX = max(bb.maxX, x)
                    bb.maxY = max(bb.maxY, y)
                    bboxes[cid] = bb
                } else {
                    bboxes[cid] = (x, y, x, y)
                }
            }
        }

        // Filter out clusters with too few visible pixels (< 0.05% of viewport)
        let minPixelCount = max(50, (width * height) / 2000)
        let visibleClusterIDs = bboxes.keys.filter { (pixelCounts[$0] ?? 0) >= minPixelCount }
        let discardedCount = bboxes.count - visibleClusterIDs.count
        
        print("[CLIP-DEBUG] Found \(bboxes.count) cluster bounding boxes, \(discardedCount) discarded (< \(minPixelCount) px)")
        for cid in visibleClusterIDs.sorted() {
            let bb = bboxes[cid]!
            let w = bb.maxX - bb.minX + 1
            let h = bb.maxY - bb.minY + 1
            let px = pixelCounts[cid] ?? 0
            print("[CLIP-DEBUG]   Cluster \(cid): bbox=(\(bb.minX),\(bb.minY))-(\(bb.maxX),\(bb.maxY)) size=\(w)x\(h) pixels=\(px)")
        }
        Self.log.info("Found \(visibleClusterIDs.count) visible clusters (\(discardedCount) too small), encoding...")

        // Status: found clusters
        DispatchQueue.main.async { [weak self] in
            self?.onStatusUpdate?("Found \(visibleClusterIDs.count) visible clusters, cropping…")
        }

        var newIDs: [Int32] = []
        var newFeatures: [[Float]] = []

        let sortedClusterIDs = visibleClusterIDs.sorted()
        let totalClusters = sortedClusterIDs.count
        encodingProgress = (0, totalClusters)

        for (index, cid) in sortedClusterIDs.enumerated() {
            guard let bb = bboxes[cid] else { continue }

            let cropW = bb.maxX - bb.minX + 1
            let cropH = bb.maxY - bb.minY + 1
            guard cropW > 2 && cropH > 2 else {
                print("[CLIP-DEBUG]   Skipping cluster \(cid): crop too small (\(cropW)x\(cropH))")
                // Still count as progress even if skipped
                encodingProgress = (index + 1, totalClusters)
                DispatchQueue.main.async { [weak self] in
                    self?.onStatusUpdate?("Encoding \(index + 1) / \(totalClusters)")
                }
                continue
            }

            // Extract the crop(s) from BGRA data → RGB CGImage
            var cropImageUnmasked: CGImage?
            var cropImageMasked: CGImage?

            // Always extract unmasked crop
            cropImageUnmasked = extractCrop(rgbData: rgbData, width: width, height: height,
                                            x: bb.minX, y: bb.minY, w: cropW, h: cropH,
                                            clusterData: nil, clusterID: cid)

            // Extract masked crop if needed
            if averageMaskedAndUnmasked || useMaskedCrops {
                cropImageMasked = extractCrop(rgbData: rgbData, width: width, height: height,
                                              x: bb.minX, y: bb.minY, w: cropW, h: cropH,
                                              clusterData: clusterData, clusterID: cid)
            }

            // Encode with CoreML
            print("[CLIP-DEBUG]   Encoding cluster \(cid) (\(index+1)/\(totalClusters)) crop=\(cropW)x\(cropH)")

            if averageMaskedAndUnmasked {
                // Run both modes and average
                var featuresUnmasked: [Float]? = nil
                var featuresMasked: [Float]? = nil

                if let img = cropImageUnmasked {
                    featuresUnmasked = encodeImage(img)
                }
                if let img = cropImageMasked {
                    featuresMasked = encodeImage(img)
                }

                if let fU = featuresUnmasked, let fM = featuresMasked {
                    // Average the two feature vectors
                    var averaged = [Float](repeating: 0, count: Self.featureDim)
                    for i in 0..<Self.featureDim {
                        averaged[i] = (fU[i] + fM[i]) * 0.5
                    }
                    // Re-normalize after averaging
                    var normSum: Float = 0
                    for x in averaged { normSum += x * x }
                    let norm = sqrt(normSum) + 1e-8
                    for i in averaged.indices { averaged[i] /= norm }

                    newIDs.append(cid)
                    newFeatures.append(averaged)
                    print("[CLIP-DEBUG]   Cluster \(cid) encoded (averaged), unmasked norm=\(fU.reduce(0) { $0 + $1*$1 }), masked norm=\(fM.reduce(0) { $0 + $1*$1 })")
                } else if let fU = featuresUnmasked {
                    // Fallback to unmasked only
                    newIDs.append(cid)
                    newFeatures.append(fU)
                    print("[CLIP-DEBUG]   Cluster \(cid) encoded (unmasked fallback)")
                } else if let fM = featuresMasked {
                    // Fallback to masked only
                    newIDs.append(cid)
                    newFeatures.append(fM)
                    print("[CLIP-DEBUG]   Cluster \(cid) encoded (masked fallback)")
                } else {
                    print("[CLIP-DEBUG]   Cluster \(cid) encoding FAILED (both modes)")
                }
            } else {
                // Original behavior: use whichever crop was requested
                let cropImage = useMaskedCrops ? cropImageMasked : cropImageUnmasked
                if let features = cropImage.flatMap({ encodeImage($0) }) {
                    newIDs.append(cid)
                    newFeatures.append(features)
                    print("[CLIP-DEBUG]   Cluster \(cid) encoded successfully, feature norm=\(features.reduce(0) { $0 + $1*$1 })")
                } else {
                    print("[CLIP-DEBUG]   Cluster \(cid) encoding FAILED")
                }
            }

            // Update progress
            encodingProgress = (index + 1, totalClusters)
            DispatchQueue.main.async { [weak self] in
                self?.onStatusUpdate?("Encoding \(index + 1) / \(totalClusters)")
            }
        }

        featureStateLock.lock()
        clusterIDsStorage = newIDs
        clusterFeaturesStorage = newFeatures
        featureStateLock.unlock()

        print("[CLIP-DEBUG] Encoding complete: \(newIDs.count) clusters encoded out of \(totalClusters) total")
        DispatchQueue.main.async { [weak self] in
            self?.onStatusUpdate?("Done — \(newIDs.count) clusters encoded")
        }
        Self.log.info("Encoded \(newIDs.count) cluster features (dim=\(Self.featureDim))")
    }

    // MARK: - Query

    /// Compute cosine similarities between text features and stored cluster features.
    /// Returns [(clusterID, similarity)] sorted by descending similarity.
    func query(text: String) -> [(clusterID: Int32, similarity: Float)] {
        guard let textFeats = encodeText(text) else { return [] }

        let featureSnapshot: [[Float]]
        let idSnapshot: [Int32]
        featureStateLock.lock()
        featureSnapshot = clusterFeaturesStorage
        idSnapshot = clusterIDsStorage
        featureStateLock.unlock()

        guard !featureSnapshot.isEmpty, !idSnapshot.isEmpty else { return [] }

        var results: [(Int32, Float)] = []
        for (i, imgFeats) in featureSnapshot.enumerated() {
            guard i < idSnapshot.count else { break }
            let sim = dot(textFeats, imgFeats)
            results.append((idSnapshot[i], sim))
        }
        results.sort { $0.1 > $1.1 }
        return results
    }

    // MARK: - Helpers

    private func l2Normalize(_ v: inout [Float]) {
        var sumSq: Float = 0
        for x in v { sumSq += x * x }
        let norm = sqrt(sumSq) + 1e-8
        for i in v.indices { v[i] /= norm }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        for i in 0..<min(a.count, b.count) {
            result += a[i] * b[i]
        }
        return result
    }

    /// Extract a bounding-box crop from BGRA pixel data and return as an RGB CGImage.
    /// If `clusterData` is provided, pixels not belonging to `clusterID` are set to white (masked out).
    private func extractCrop(rgbData: [UInt8], width: Int, height: Int,
                              x: Int, y: Int, w: Int, h: Int,
                              clusterData: [Int32]? = nil, clusterID: Int32 = -1) -> CGImage? {
        // Use RGBA (4 bytes/pixel) — CGBitmapContext doesn't support 3-byte RGB
        var cropRGBA = [UInt8](repeating: 255, count: w * h * 4)
        for row in 0..<h {
            for col in 0..<w {
                let srcX = x + col
                let srcY = y + row
                let srcIdx = (srcY * width + srcX) * 4
                let dstIdx = (row * w + col) * 4

                // If masking is enabled, check if this pixel belongs to the cluster
                if let clusterData = clusterData {
                    let pixelCluster = clusterData[srcY * width + srcX]
                    if pixelCluster != clusterID {
                        // Masked out → white background
                        cropRGBA[dstIdx + 0] = 255
                        cropRGBA[dstIdx + 1] = 255
                        cropRGBA[dstIdx + 2] = 255
                        cropRGBA[dstIdx + 3] = 255
                        continue
                    }
                }

                // BGRA → RGBX
                cropRGBA[dstIdx + 0] = rgbData[srcIdx + 2] // R
                cropRGBA[dstIdx + 1] = rgbData[srcIdx + 1] // G
                cropRGBA[dstIdx + 2] = rgbData[srcIdx + 0] // B
                cropRGBA[dstIdx + 3] = 255                  // A (unused filler)
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: &cropRGBA,
                                       width: w, height: h,
                                       bitsPerComponent: 8, bytesPerRow: w * 4,
                                       space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        return context.makeImage()
    }

    /// Resize a CGImage to targetSize × targetSize
    private func resizeCGImage(_ image: CGImage, to size: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil,
                                       width: size, height: size,
                                       bitsPerComponent: 8, bytesPerRow: size * 4,
                                       space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return context.makeImage()
    }

    /// Create a CVPixelBuffer from a CGImage (for CoreML imageType input)
    private func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                          width, height,
                                          kCVPixelFormatType_32BGRA,
                                          attrs as CFDictionary,
                                          &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        guard let context = CGContext(data: baseAddress,
                                       width: width, height: height,
                                       bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                       space: CGColorSpaceCreateDeviceRGB(),
                                       bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else {
            return nil
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

#endif // os(iOS) || os(macOS)
