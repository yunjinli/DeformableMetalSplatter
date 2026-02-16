# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deformable MetalSplatter is an Apple platform app (iOS, macOS, visionOS) for rendering deformable 3D Gaussian Splats using Metal. It extends the original MetalSplatter with support for dynamic (time-varying) splats and instance segmentation via TRASE.

## Build Commands

```bash
# Build the Swift package
swift build

# Run all tests
swift test

# Run a specific test
swift test --filter PLYIOTests

# Run a specific test target
swift test --target PLYIOTests
```

The SampleApp directory contains an Xcode project (MetalSplatter_SampleApp.xcodeproj) for running the app on device/simulator.

## Architecture

### Swift Package Structure

- **PLYIO** (`PLYIO/`): Pure Swift library for reading/writing PLY (Polygon File Format) files. Supports both ASCII and binary formats with a delegate-based streaming parser.

- **SplatIO** (`SplatIO/`): Handles splat scene I/O. Reads/writes `.ply` files and `.splat` binary format. Built on top of PLYIO.

- **MetalSplatter** (`MetalSplatter/`): Core rendering library. Contains:
  - `SplatRenderer.swift`: Main renderer class managing Gaussian splat rendering
  - `DeformGraphSystem.swift`: Handles deformation network and time-based transformations
  - Metal shaders in `Resources/`:
    - `Deform.metal`: Gaussian deformation (rotation, scale, position deltas)
    - `SplatProcessing.metal`: Pre-processing splats
    - `SingleStageRenderPath.metal` / `MultiStageRenderPath.metal`: Rendering pipelines

- **SampleApp** (`SampleApp/`): SwiftUI/MetalKit app demonstrating the renderer. Contains:
  - `Model/`: CLIPService for text-based clustering, ModelRenderer
  - `Scene/`: VisionSceneRenderer, MetalKitSceneRenderer
  - `App/`: Main app entry point

- **SplatConverter** (`SplatConverter/`): Command-line tool for converting between splat formats.

### Python Data Pipeline

The project uses Python scripts for training data preparation:

- `export_deform_weights.py`: Exports PyTorch DeformNetwork weights to binary format for Metal consumption. Includes a `DeformNetwork` MLP class (D=8, W=256) that predicts position, rotation, and scale deltas.

- `run_deformation.py`: Applies deformation at a specific time t (0-1) to a PLY file. Run with `--smooth` to apply only position deltas (skips rotation/scale for noisy dense recordings).

- `export_clusters_bin.py`: Exports TRASE instance segmentation clusters.

- `encode_clusters.py`, `query_clusters.py`: For CLIP-based cluster encoding and querying.

- `convert_mobileclip_coreml.py`: Converts MobileCLIP models to CoreML for on-device feature extraction.

### Key Data Files

- `weights.bin`: Deformation network weights (exported by export_deform_weights.py)
- `clusters.bin`: Instance segmentation clusters
- `point_cloud.ply`: Base Gaussian splat positions/properties
- `deformed_*.ply`: Time-deformed PLY outputs

### CoreML Models

Located in `coreml_models/`:
- MobileCLIP image and text encoders for semantic clustering
- Models exported from HuggingFace transformers

## Development Notes

- The Metal shaders use column-major matrix layout (Metal default). Recent fix in `Deform.metal` corrected covariance matrix construction to use `R * S² * R^T` instead of transposed version.

- Smooth mode in deformation (`--smooth` flag) skips rotation and scale deltas, keeping canonical values to reduce noise.

- The app expects data files (weights.bin, clusters.bin, point_cloud.ply) in the same directory selected at startup.
