"""
Encode all cluster crops with MobileCLIP and save per-cluster image features.

Supports two backends:
  - coreml  (default on macOS) – uses our converted MobileCLIP2-S0 CoreML .mlpackage models
  - pytorch – uses open_clip + mobileclip (requires GPU-capable env)

Usage:
    python encode_clusters.py <crops_folder> [--output features.npz] [--backend coreml]

Example:
    python encode_clusters.py my_cluster_crops --output cluster_features.npz
"""

import numpy as np
import argparse
import os
import glob
import re
from tqdm import tqdm


# ── CoreML backend ──────────────────────────────────────────────────────────

def _encode_coreml(crops_folder, output_path, model_path):
    import coremltools as ct
    from PIL import Image

    # Load CoreML image encoder
    print(f"Loading CoreML image encoder from {model_path} ...")
    mlmodel = ct.models.MLModel(model_path)

    # Inspect model to find input/output names
    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    print(f"  Input: '{input_name}', Output: '{output_name}'")

    # Find all cluster crop images
    crop_files = sorted(glob.glob(os.path.join(crops_folder, "cluster_*.png")))
    if not crop_files:
        print(f"No cluster_*.png files found in {crops_folder}")
        return

    print(f"Found {len(crop_files)} cluster crops.")

    cluster_ids = []
    features_list = []

    for crop_file in tqdm(crop_files):
        basename = os.path.basename(crop_file)
        match = re.match(r"cluster_(\d+)\.png", basename)
        if not match:
            print(f"  Skipping {basename} (unexpected name format)")
            continue

        cid = int(match.group(1))
        cluster_ids.append(cid)

        # Load and resize to 256x256 (MobileCLIP-S0 expected input)
        img = Image.open(crop_file).convert("RGB").resize((256, 256))

        # Run CoreML prediction
        pred = mlmodel.predict({input_name: img})
        feat = np.array(pred[output_name]).flatten()

        # L2 normalize
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        features_list.append(feat.astype(np.float32))

    return cluster_ids, features_list


# ── PyTorch backend ─────────────────────────────────────────────────────────

def _encode_pytorch(crops_folder, output_path, model_name, pretrained):
    import torch
    import open_clip
    from PIL import Image
    from mobileclip.modules.common.mobileone import reparameterize_model

    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    print(f"Loading model {model_name} (pretrained={pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, **model_kwargs
    )
    model.eval()
    model = reparameterize_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    crop_files = sorted(glob.glob(os.path.join(crops_folder, "cluster_*.png")))
    if not crop_files:
        print(f"No cluster_*.png files found in {crops_folder}")
        return None, None

    print(f"Found {len(crop_files)} cluster crops.")

    cluster_ids = []
    features_list = []

    for crop_file in tqdm(crop_files):
        basename = os.path.basename(crop_file)
        match = re.match(r"cluster_(\d+)\.png", basename)
        if not match:
            print(f"  Skipping {basename} (unexpected name format)")
            continue

        cid = int(match.group(1))
        cluster_ids.append(cid)

        img = Image.open(crop_file).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        features_list.append(features.cpu().numpy().squeeze(0))

    return cluster_ids, features_list


# ── Main ────────────────────────────────────────────────────────────────────

def encode_clusters(crops_folder, output_path, backend="coreml",
                    model_path=None, model_name="MobileCLIP2-S0", pretrained="dfndr2b"):

    if backend == "coreml":
        if model_path is None:
            # Default path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "coreml_models", "MobileCLIPImageEncoder.mlpackage")
        cluster_ids, features_list = _encode_coreml(crops_folder, output_path, model_path)
    else:
        cluster_ids, features_list = _encode_pytorch(crops_folder, output_path, model_name, pretrained)

    if cluster_ids is None or len(cluster_ids) == 0:
        print("No clusters were encoded.")
        return

    cluster_ids = np.array(cluster_ids, dtype=np.int32)
    features_array = np.stack(features_list, axis=0)  # (N, D)

    np.savez(output_path,
             cluster_ids=cluster_ids,
             features=features_array,
             model_name=model_name if backend == "pytorch" else "MobileCLIP2-S0",
             pretrained=pretrained if backend == "pytorch" else "dfndr2b")

    print(f"\nSaved {len(cluster_ids)} cluster features to {output_path}")
    print(f"  Feature shape: {features_array.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode cluster crops with MobileCLIP.")
    parser.add_argument("crops_folder", help="Folder containing cluster_*.png crops")
    parser.add_argument("--output", default="cluster_features.npz",
                        help="Output .npz file (default: cluster_features.npz)")
    parser.add_argument("--backend", default="coreml", choices=["coreml", "pytorch"],
                        help="Inference backend (default: coreml)")
    parser.add_argument("--model_path", default=None,
                        help="Path to CoreML .mlpackage (default: coreml_models/MobileCLIPImageEncoder.mlpackage)")
    parser.add_argument("--model_name", default="MobileCLIP2-S0",
                        help="PyTorch model name (default: MobileCLIP2-S0)")
    parser.add_argument("--pretrained", default="dfndr2b",
                        help="PyTorch pretrained tag (default: dfndr2b)")
    args = parser.parse_args()

    encode_clusters(args.crops_folder, args.output, args.backend,
                    args.model_path, args.model_name, args.pretrained)
