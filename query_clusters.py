"""
Query cluster features with a text prompt and visualize matching clusters.

Usage:
    python query_clusters.py <features.npz> <rgb.raw> <cluster.bin> <text_prompt> [--output result.png] [--threshold 0.2] [--top_k 3]

Example:
    python query_clusters.py cluster_features.npz \
        /Users/cedric/Documents/capture_123_rgb.raw \
        /Users/cedric/Documents/capture_123_ids.bin \
        "a cookie" \
        --output result.png --top_k 3
"""

import numpy as np
import open_clip
import coremltools as ct
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.ndimage
import argparse
import os


def load_frame(rgb_path, cluster_path):
    """Load RGB and cluster data from raw files."""
    meta_path = rgb_path.replace("_rgb.raw", "_meta.txt")
    width = 0
    height = 0

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if "Width:" in line:
                    width = int(line.split(":")[1].strip())
                elif "Height:" in line:
                    height = int(line.split(":")[1].strip())

    if width == 0 or height == 0:
        raise ValueError(f"Could not determine dimensions from {meta_path}")

    rgb_data = np.fromfile(rgb_path, dtype=np.uint8).reshape((height, width, 4))
    # BGRA -> RGBA
    rgb_img = rgb_data[..., [2, 1, 0, 3]]

    cluster_img = np.fromfile(cluster_path, dtype=np.int32).reshape((height, width))

    # Hole filling (same as visualize_clusters.py)
    invalid_mask = (cluster_img == -1)
    if np.any(invalid_mask):
        unique_ids = np.unique(cluster_img)
        for cid in unique_ids:
            if cid == -1:
                continue
            cluster_mask = (cluster_img == cid)
            filled_mask = scipy.ndimage.binary_fill_holes(cluster_mask)
            holes_to_fill = filled_mask & (cluster_img == -1)
            if np.any(holes_to_fill):
                cluster_img[holes_to_fill] = cid

        invalid_mask = (cluster_img == -1)
        if np.any(invalid_mask):
            distances, indices = scipy.ndimage.distance_transform_edt(
                invalid_mask, return_distances=True, return_indices=True
            )
            cluster_img = cluster_img[tuple(indices)]
            cluster_img[distances > 20] = -1

    return rgb_img, cluster_img, width, height


def encode_text_coreml(text_prompt, text_model_path=None):
    """Encode text using our converted MobileCLIP2-S0 CoreML text encoder."""
    if text_model_path == None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        text_model_path = os.path.join(script_dir, "coreml_models", "MobileCLIP2TextEncoder.mlpackage")

    print(f"Loading CoreML text encoder from {text_model_path} ...")
    mlmodel = ct.models.MLModel(text_model_path)

    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    print(f"  Input: '{input_name}', Output: '{output_name}'")

    # Use open_clip tokenizer (lightweight, does not load full model weights)
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
    tokens = tokenizer([text_prompt]).numpy().astype(np.int32)  # (1, 77)
    print(f"  Tokenized -> {(tokens[0] != 0).sum()} non-zero tokens")

    # Run CoreML text encoder
    pred = mlmodel.predict({input_name: tokens})
    feat = np.array(pred[output_name]).flatten().astype(np.float32)

    # Already L2-normalized by the wrapper, but ensure it
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


def query_clusters(features_path, rgb_path, cluster_path, text_prompt,
                   output_path, text_model_path=None,
                   threshold=None, top_k=None):
    # Load features
    data = np.load(features_path)
    cluster_ids = data["cluster_ids"]
    features = data["features"]  # (N, D)

    print(f"Loaded {len(cluster_ids)} cluster features (dim={features.shape[1]})")
    print(f'Query: "{text_prompt}"')

    # Encode text with CoreML (MobileCLIP2-S0)
    text_features_np = encode_text_coreml(text_prompt, text_model_path)

    # Compute cosine similarities
    # features is already L2-normalized from encode_clusters.py
    similarities = features @ text_features_np  # (N,)

    # Print all similarities
    print("\n--- Cosine Similarities ---")
    sorted_indices = np.argsort(-similarities)
    for idx in sorted_indices:
        cid = cluster_ids[idx]
        sim = similarities[idx]
        print(f"  Cluster {cid:4d}: {sim:.4f}")

    # Select clusters
    if top_k is not None:
        selected_indices = sorted_indices[:top_k]
        selected_ids = set(cluster_ids[selected_indices])
        print(f"\nSelected top-{top_k} clusters: {selected_ids}")
    elif threshold is not None:
        selected_mask = similarities >= threshold
        selected_ids = set(cluster_ids[selected_mask])
        print(f"\nSelected {len(selected_ids)} clusters with similarity >= {threshold}: {selected_ids}")
    else:
        # Default: top 1
        selected_indices = sorted_indices[:1]
        selected_ids = set(cluster_ids[selected_indices])
        print(f"\nSelected top cluster: {selected_ids}")

    if len(selected_ids) == 0:
        print("No clusters selected! Try lowering the threshold or increasing top_k.")
        return

    # Load frame
    print("Loading frame data...")
    rgb_img, cluster_img, width, height = load_frame(rgb_path, cluster_path)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # 1. Original RGB
    axes[0].set_title("Original RGB")
    axes[0].imshow(rgb_img)
    axes[0].axis('off')

    # 2. Selected clusters highlighted
    # Darken non-selected regions, keep selected bright
    darkened = rgb_img.copy().astype(np.float32)
    selected_mask_2d = np.zeros((height, width), dtype=bool)
    for cid in selected_ids:
        selected_mask_2d |= (cluster_img == int(cid))

    # Darken non-selected
    darkened[~selected_mask_2d] *= 0.3
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)

    # Add colored border around selected regions
    dilated = scipy.ndimage.binary_dilation(selected_mask_2d, iterations=5)
    outline = dilated & ~selected_mask_2d
    darkened[outline] = [255, 50, 50, 255]  # Red outline

    axes[1].set_title(f"Selected: \"{text_prompt}\"")
    axes[1].imshow(darkened)
    axes[1].axis('off')

    # 3. Cluster overlay with selected highlighted
    np.random.seed(42)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    unique_clusters = np.unique(cluster_img)

    for cid in unique_clusters:
        if cid == -1:
            continue
        mask = (cluster_img == cid)
        color = np.random.randint(50, 255, 3)

        if cid in selected_ids:
            # Bright, full opacity for selected
            overlay[mask] = [color[0], color[1], color[2], 200]
        else:
            # Very faint for non-selected
            overlay[mask] = [color[0], color[1], color[2], 40]

    axes[2].set_title("Cluster Map (selected highlighted)")
    axes[2].imshow(rgb_img)
    axes[2].imshow(overlay)
    axes[2].axis('off')

    # Add legend for selected clusters
    legend_patches = []
    for idx in sorted_indices:
        cid = cluster_ids[idx]
        sim = similarities[idx]
        if cid in selected_ids:
            legend_patches.append(
                mpatches.Patch(color='red', label=f"Cluster {cid} (sim={sim:.3f})")
            )
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc='lower right', fontsize=8)

    plt.suptitle(f"Text Query: \"{text_prompt}\"", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved result to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query cluster features with text and visualize.")
    parser.add_argument("features", help="Path to cluster_features.npz")
    parser.add_argument("rgb", help="Path to .raw RGB file")
    parser.add_argument("cluster", help="Path to .bin Cluster ID file")
    parser.add_argument("text", help="Text prompt to search for")
    parser.add_argument("--output", default="query_result.png", help="Output PNG (default: query_result.png)")
    parser.add_argument("--text_model_path", default=None,
                        help="Path to CoreML text .mlpackage (default: coreml_models/MobileCLIP2TextEncoder.mlpackage)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Select all clusters with similarity >= threshold")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Select top-k clusters by similarity")
    args = parser.parse_args()
    
    query_clusters(
        args.features, args.rgb, args.cluster, args.text,
        args.output, args.text_model_path,
        args.threshold, args.top_k
    )
