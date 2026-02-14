import Foundation

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL, useFP16: Bool)

    var description: String {
        switch self {
        case .gaussianSplat(let url, let useFP16):
            "Gaussian Splat: \(url.path) (\(useFP16 ? "FP16" : "FP32"))"
        }
    }
    
    var useFP16: Bool {
        switch self {
        case .gaussianSplat(_, let useFP16):
            return useFP16
        }
    }
}
