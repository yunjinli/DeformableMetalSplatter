#!/usr/bin/env python3
"""
Generate a deformation mask for a deformable splatting scene.

This script computes which splats move between t=0 (baseline) and a target timestep.
It outputs a binary mask where 1.0 = moving splat, 0.0 = static splat.

The mask can be used in the Metal app to skip rendering computation for static splats.

Two modes are supported:
1. Per-splat: Mark individual splats as moving/static based on position threshold
2. Per-cluster: Mark all splats in a cluster as moving if any splat in that cluster moves
"""

import torch
import numpy as np
import argparse
import sys
import os
import struct
import json

from plyfile import PlyData
# Import DeformNetwork from local file
from export_deform_weights import DeformNetwork


def load_clusters_bin(clusters_path, expected_count):
    """Load cluster IDs from clusters.bin file.

    Returns:
        cluster_ids: numpy array of shape (num_points,) with cluster IDs (UInt32)
    """
    with open(clusters_path, 'rb') as f:
        # Read header
        magic = f.read(4).decode('utf-8')
        if magic != 'CLST':
            raise ValueError(f"Invalid clusters.bin: expected 'CLST' magic, got '{magic}'")

        version = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<I', f.read(4))[0]

        print(f"Clusters.bin: version={version}, count={count}")

        if count != expected_count:
            print(f"Warning: clusters.bin count ({count}) != PLY point count ({expected_count})")

        # Read cluster IDs
        cluster_ids = np.frombuffer(f.read(count * 4), dtype=np.uint32)

    return cluster_ids


def compute_deformation_deltas(ply_path, model_path, t, smooth=False):
    """Compute position, rotation, and scale deltas at a given time t."""
    print(f"Computing deltas at t={t} (smooth={smooth})")

    # 1. Load PLY
    plydata = PlyData.read(ply_path)
    if 'vertex' not in plydata:
        print("Error: PLY file does not contain 'vertex' element")
        sys.exit(1)

    vertex = plydata['vertex']

    # Extract positions
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # Extract scales
    s0 = np.array(vertex['scale_0'])
    s1 = np.array(vertex['scale_1'])
    s2 = np.array(vertex['scale_2'])
    scales = np.stack([s0, s1, s2], axis=1).astype(np.float32)

    # Extract rotations (w, x, y, z)
    r0 = np.array(vertex['rot_0'])
    r1 = np.array(vertex['rot_1'])
    r2 = np.array(vertex['rot_2'])
    r3 = np.array(vertex['rot_3'])
    rotations = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)

    num_points = positions.shape[0]
    print(f"Loaded {num_points} points")

    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeformNetwork(D=8, W=256, verbose=False).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model.eval()

    # 3. Compute deltas in batches
    t_tensor = torch.full((num_points, 1), t, dtype=torch.float32, device=device)
    positions_tensor = torch.from_numpy(positions).to(device)

    batch_size = 16384 * 4
    d_xyz_list = []
    d_rot_list = []
    d_scale_list = []

    print("Running inference...")
    with torch.no_grad():
        for i in range(0, num_points, batch_size):
            batch_pos = positions_tensor[i:i+batch_size]
            batch_t = t_tensor[i:i+batch_size]

            d_x, d_r, d_s = model(batch_pos, batch_t)

            d_xyz_list.append(d_x.cpu().numpy())
            d_rot_list.append(d_r.cpu().numpy())
            d_scale_list.append(d_s.cpu().numpy())

    d_xyz = np.concatenate(d_xyz_list, axis=0)
    d_rot = np.concatenate(d_rot_list, axis=0)
    d_scale = np.concatenate(d_scale_list, axis=0)

    return positions, scales, rotations, d_xyz, d_rot, d_scale, smooth


def generate_deformation_mask(ply_path, model_path, output_path, threshold=0.001,
                               smooth=False, max_t=1.0, num_samples=10):
    """
    Generate a continuous deformation magnitude mask.

    For each splat, stores the maximum position displacement (L2 norm) observed
    across all sampled timesteps. This allows the app to apply dynamic thresholding
    at runtime for a user-controlled quality/performance tradeoff.

    Args:
        ply_path: Input PLY file with Gaussian splat data
        model_path: Path to deform.pth model weights
        output_path: Output .bin file for the mask
        threshold: Only used for stats display, not for masking
        smooth: Whether to use smooth mode (position deltas only)
        max_t: Maximum time to sample (default 1.0)
        num_samples: Number of timesteps to sample between 0 and max_t
    """
    print(f"Generating continuous deformation mask: max_t={max_t}, samples={num_samples}")

    # Compute deltas at t=0 (baseline)
    print("\n=== Computing baseline at t=0 ===")
    positions_0, scales_0, rotations_0, d_xyz_0, d_rot_0, d_scale_0, smooth = \
        compute_deformation_deltas(ply_path, model_path, t=0.0, smooth=smooth)

    # Get baseline positions (canonical + delta at t=0)
    baseline_positions = positions_0 + d_xyz_0

    # Initialize mask: stores max displacement magnitude per splat
    num_points = positions_0.shape[0]
    mask = np.zeros(num_points, dtype=np.float32)

    # Sample timesteps from 0 to max_t (excluding 0 since it's the baseline)
    timesteps = np.linspace(0, max_t, num_samples + 1)[1:]  # Skip t=0

    for t in timesteps:
        print(f"\n=== Computing deltas at t={t:.3f} ===")
        _, _, _, d_xyz_t, _, _, _ = compute_deformation_deltas(ply_path, model_path, t=t, smooth=smooth)

        # Compute deformed positions at time t
        deformed_positions = positions_0 + d_xyz_t

        # Compute position difference from baseline
        diff = np.linalg.norm(deformed_positions - baseline_positions, axis=1)

        # Keep the maximum displacement across all timesteps
        mask = np.maximum(mask, diff)

        # Progress update
        moving_count = np.sum(mask > threshold)
        print(f"t={t:.3f}: {moving_count}/{num_points} splats above ref threshold {threshold} ({100*moving_count/num_points:.2f}%)")

    # Final statistics
    nonzero_mask = mask[mask > 1e-8]
    print(f"\n=== Final mask statistics (continuous) ===")
    print(f"Total splats: {num_points}")
    print(f"Max displacement: {np.max(mask):.6f}")
    if len(nonzero_mask) > 0:
        print(f"Mean displacement (nonzero): {np.mean(nonzero_mask):.6f}")
        print(f"Median displacement (nonzero): {np.median(nonzero_mask):.6f}")
        for pct in [50, 75, 90, 95, 99]:
            val = np.percentile(nonzero_mask, pct)
            count_above = np.sum(mask > val)
            print(f"  P{pct}: {val:.6f} ({count_above} splats above = {100*count_above/num_points:.2f}%)")

    # Save mask (continuous float32 values)
    print(f"\nSaving continuous mask to {output_path}")
    mask.tofile(output_path)
    print(f"Saved {num_points} float32 values (continuous magnitudes)")

    # Save summary
    summary_path = output_path + '.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Deformation Mask Summary (Continuous Per-Splat)\n")
        f.write(f"===============================================\n")
        f.write(f"PLY: {ply_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Max t: {max_t}\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Total splats: {num_points}\n")
        f.write(f"Max displacement: {np.max(mask):.6f}\n")
        if len(nonzero_mask) > 0:
            f.write(f"Mean displacement (nonzero): {np.mean(nonzero_mask):.6f}\n")
            f.write(f"Median displacement (nonzero): {np.median(nonzero_mask):.6f}\n")
            for pct in [50, 75, 90, 95, 99]:
                val = np.percentile(nonzero_mask, pct)
                count_above = np.sum(mask > val)
                f.write(f"P{pct}: {val:.6f} ({count_above}/{num_points} above = {100*count_above/num_points:.2f}%)\n")
    print(f"Saved summary to {summary_path}")

    # Calculate recommended percentile and save config
    static_count = np.sum(mask <= threshold)
    recommended_percentile = float((static_count / num_points) * 100.0)
    
    config_path = output_path.replace('.bin', '.json')
    if config_path == output_path:
        config_path += '.json'
    
    with open(config_path, 'w') as f:
        json.dump({"recommended_percentile": recommended_percentile}, f, indent=4)
    print(f"Saved config to {config_path} (recommended_percentile: {recommended_percentile:.1f}%)")

    return mask


def generate_cluster_based_mask(ply_path, model_path, clusters_path, output_path, threshold=0.001,
                                smooth=False, max_t=1.0, num_samples=10):
    """
    Generate a deformation mask at the cluster level.

    A cluster is marked as moving (1.0) if ANY splat in that cluster moves beyond the threshold.
    All splats in a moving cluster get mask=1.0, all splats in static clusters get mask=0.0.

    Args:
        ply_path: Input PLY file with Gaussian splat data
        model_path: Path to deform.pth model weights
        clusters_path: Path to clusters.bin file
        output_path: Output .bin file for the mask
        threshold: Position change threshold to consider a splat as moving
        smooth: Whether to use smooth mode
        max_t: Maximum time to sample
        num_samples: Number of timesteps to sample
    """
    print(f"Generating CLUSTER-BASED deformation mask: threshold={threshold}, max_t={max_t}, samples={num_samples}")

    # 1. Load PLY
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    s0 = np.array(vertex['scale_0'])
    s1 = np.array(vertex['scale_1'])
    s2 = np.array(vertex['scale_2'])
    scales = np.stack([s0, s1, s2], axis=1).astype(np.float32)

    r0 = np.array(vertex['rot_0'])
    r1 = np.array(vertex['rot_1'])
    r2 = np.array(vertex['rot_2'])
    r3 = np.array(vertex['rot_3'])
    rotations = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)

    num_points = positions.shape[0]
    print(f"Loaded {num_points} points from PLY")

    # 2. Load clusters
    print(f"Loading clusters from {clusters_path}")
    cluster_ids = load_clusters_bin(clusters_path, num_points)
    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    print(f"Found {num_clusters} unique clusters")

    # 3. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeformNetwork(D=8, W=256, verbose=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Compute deltas at t=0 (baseline)
    print("\n=== Computing baseline at t=0 ===")
    device_tensor = torch.device(device)
    t_tensor_0 = torch.full((num_points, 1), 0.0, dtype=torch.float32, device=device)
    positions_tensor = torch.from_numpy(positions).to(device)

    batch_size = 16384 * 4
    d_xyz_0_list = []

    with torch.no_grad():
        for i in range(0, num_points, batch_size):
            batch_pos = positions_tensor[i:i+batch_size]
            batch_t = t_tensor_0[i:i+batch_size]
            d_x, _, _ = model(batch_pos, batch_t)
            d_xyz_0_list.append(d_x.cpu().numpy())

    d_xyz_0 = np.concatenate(d_xyz_0_list, axis=0)
    baseline_positions = positions + d_xyz_0

    # 5. Track which clusters are moving
    cluster_is_moving = set()

    # Sample timesteps
    timesteps = np.linspace(0, max_t, num_samples + 1)[1:]

    for t in timesteps:
        print(f"\n=== Computing deltas at t={t:.3f} ===")

        t_tensor = torch.full((num_points, 1), t, dtype=torch.float32, device=device)
        d_xyz_t_list = []

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                batch_pos = positions_tensor[i:i+batch_size]
                batch_t = t_tensor[i:i+batch_size]
                d_x, _, _ = model(batch_pos, batch_t)
                d_xyz_t_list.append(d_x.cpu().numpy())

        d_xyz_t = np.concatenate(d_xyz_t_list, axis=0)
        deformed_positions = positions + d_xyz_t

        # Compute position difference from baseline
        diff = np.linalg.norm(deformed_positions - baseline_positions, axis=1)

        # Find splats that are moving
        moving_splats = diff > threshold

        # Find clusters that have at least one moving splat
        moving_cluster_ids = np.unique(cluster_ids[moving_splats])
        cluster_is_moving.update(moving_cluster_ids.tolist())

        print(f"t={t:.3f}: {len(moving_cluster_ids)}/{num_clusters} clusters moving")

    # 6. Generate mask based on cluster membership
    mask = np.zeros(num_points, dtype=np.float32)
    for cluster_id in cluster_is_moving:
        mask[cluster_ids == cluster_id] = 1.0

    # Final statistics
    final_moving = np.sum(mask)
    moving_clusters = len(cluster_is_moving)
    print(f"\n=== Final mask statistics (per-cluster) ===")
    print(f"Total splats: {num_points}")
    print(f"Moving splats: {final_moving} ({100*final_moving/num_points:.2f}%)")
    print(f"Static splats: {num_points - final_moving} ({100*(num_points-final_moving)/num_points:.2f}%)")
    print(f"Total clusters: {num_clusters}")
    print(f"Moving clusters: {moving_clusters} ({100*moving_clusters/num_clusters:.2f}%)")
    print(f"Static clusters: {num_clusters - moving_clusters} ({100*(num_clusters-moving_clusters)/num_clusters:.2f}%)")

    # Save mask
    print(f"\nSaving mask to {output_path}")
    mask.tofile(output_path)
    print(f"Saved {num_points} float32 values")

    # Save summary
    summary_path = output_path + '.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Deformation Mask Summary (Cluster-Based)\n")
        f.write(f"========================================\n")
        f.write(f"PLY: {ply_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Clusters: {clusters_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Max t: {max_t}\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Total splats: {num_points}\n")
        f.write(f"Moving splats: {final_moving} ({100*final_moving/num_points:.2f}%)\n")
        f.write(f"Static splats: {num_points - final_moving} ({100*(num_points-final_moving)/num_points:.2f}%)\n")
        f.write(f"Total clusters: {num_clusters}\n")
        f.write(f"Moving clusters: {moving_clusters} ({100*moving_clusters/num_clusters:.2f}%)\n")
        f.write(f"Static clusters: {num_clusters - moving_clusters} ({100*(num_clusters-moving_clusters)/num_clusters:.2f}%)\n")
    print(f"Saved summary to {summary_path}")

    # Calculate recommended percentile and save config
    static_count = np.sum(mask <= threshold)
    recommended_percentile = float((static_count / num_points) * 100.0)
    
    config_path = output_path.replace('.bin', '.json')
    if config_path == output_path:
        config_path += '.json'
    
    with open(config_path, 'w') as f:
        json.dump({"recommended_percentile": recommended_percentile}, f, indent=4)
    print(f"Saved config to {config_path} (recommended_percentile: {recommended_percentile:.1f}%)")

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a deformation mask for deformable splatting. "
                    "Compares t=0 baseline with sampled timesteps to identify moving splats."
    )
    parser.add_argument("ply_path", help="Input PLY file with Gaussian splat data")
    parser.add_argument("model_path", help="Path to deform.pth model weights")
    parser.add_argument("output_path", help="Output .bin file for the deformation mask")

    parser.add_argument("--clusters", type=str, default=None,
                        help="Path to clusters.bin file. If provided, uses cluster-based masking "
                             "(all splats in a moving cluster are marked as dynamic)")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Position change threshold to mark splat/cluster as moving (default: 0.001)")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Smooth mode: use only position deltas, skip rotation/scale")
    parser.add_argument("--max-t", type=float, default=1.0,
                        help="Maximum time to sample (default: 1.0)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of timesteps to sample (default: 10)")

    args = parser.parse_args()

    if args.threshold <= 0:
        print("Error: threshold must be positive")
        sys.exit(1)

    if not 0 < args.max_t <= 1.0:
        print("Error: max-t should be between 0 and 1")
        sys.exit(1)

    if args.clusters:
        # Cluster-based mode
        if not os.path.exists(args.clusters):
            print(f"Error: clusters file not found: {args.clusters}")
            sys.exit(1)

        generate_cluster_based_mask(
            ply_path=args.ply_path,
            model_path=args.model_path,
            clusters_path=args.clusters,
            output_path=args.output_path,
            threshold=args.threshold,
            smooth=args.smooth,
            max_t=args.max_t,
            num_samples=args.num_samples
        )
    else:
        # Per-splat mode
        generate_deformation_mask(
            ply_path=args.ply_path,
            model_path=args.model_path,
            output_path=args.output_path,
            threshold=args.threshold,
            smooth=args.smooth,
            max_t=args.max_t,
            num_samples=args.num_samples
        )
