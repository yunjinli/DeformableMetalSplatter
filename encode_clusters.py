"""
Encode all cluster crops with MobileCLIP and save per-cluster image features.

Usage:
    python encode_clusters.py <crops_folder> [--output features.npz] [--model_name MobileCLIP2-S0] [--pretrained dfndr2b]

Example:
    python encode_clusters.py my_cluster_crops --output cluster_features.npz
"""

import torch
import numpy as np
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
import argparse
import os
import glob
import re
from tqdm import tqdm


def encode_clusters(crops_folder, output_path, model_name="MobileCLIP2-S0", pretrained="dfndr2b"):
    # Setup model
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    print(f"Loading model {model_name} (pretrained={pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, **model_kwargs
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    model = reparameterize_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Find all cluster crop images
    crop_files = sorted(glob.glob(os.path.join(crops_folder, "cluster_*.png")))
    if not crop_files:
        print(f"No cluster_*.png files found in {crops_folder}")
        return

    print(f"Found {len(crop_files)} cluster crops.")

    cluster_ids = []
    features_list = []

    for crop_file in tqdm(crop_files):
        # Extract cluster ID from filename (e.g. cluster_5.png -> 5)
        basename = os.path.basename(crop_file)
        match = re.match(r"cluster_(\d+)\.png", basename)
        if not match:
            print(f"  Skipping {basename} (unexpected name format)")
            continue

        cid = int(match.group(1))
        cluster_ids.append(cid)

        # Load and preprocess
        img = Image.open(crop_file).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Encode
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        features_list.append(features.cpu().numpy().squeeze(0))
        print(f"  Encoded cluster {cid} ({basename})")

    # Save
    cluster_ids = np.array(cluster_ids, dtype=np.int32)
    features_array = np.stack(features_list, axis=0)  # (N, D)

    np.savez(output_path,
             cluster_ids=cluster_ids,
             features=features_array,
             model_name=model_name,
             pretrained=pretrained)

    print(f"\nSaved {len(cluster_ids)} cluster features to {output_path}")
    print(f"  Feature shape: {features_array.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode cluster crops with MobileCLIP.")
    parser.add_argument("crops_folder", help="Folder containing cluster_*.png crops")
    parser.add_argument("--output", default="cluster_features.npz", help="Output .npz file (default: cluster_features.npz)")
    parser.add_argument("--model_name", default="MobileCLIP2-S0", help="Model name (default: MobileCLIP2-S0)")
    parser.add_argument("--pretrained", default="dfndr2b", help="Pretrained tag (default: dfndr2b)")
    args = parser.parse_args()

    encode_clusters(args.crops_folder, args.output, args.model_name, args.pretrained)
