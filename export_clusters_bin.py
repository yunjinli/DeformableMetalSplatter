import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def export_clusters_bin(src_pt: Path, dst_bin: Path) -> None:
    cluster_data = torch.load(src_pt, weights_only=False, map_location="cpu")
    
    ids = _to_numpy(cluster_data["id"]).ravel().astype(np.uint32)
    
    # Handle colors the same way as load_clusters.py (no blind reshape)
    colors = _to_numpy(cluster_data["rgb"])
    print(f"Original rgb shape: {colors.shape}, dtype: {colors.dtype}")
    
    # Ensure (N, 3) shape - transpose if channels-first (3, N)
    if colors.ndim == 2:
        if colors.shape[0] == 3 and colors.shape[1] != 3:
            print("Detected channels-first (3, N), transposing to (N, 3)")
            colors = colors.T
    elif colors.ndim == 1:
        # Flat array, assume interleaved RGB
        colors = colors.reshape(-1, 3)
    
    # Ensure C-contiguous for correct byte ordering
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    print(f"Final colors shape: {colors.shape}, C-contiguous: {colors.flags['C_CONTIGUOUS']}")
    
    if colors.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) colors, got {colors.shape}")
    
    if ids.shape[0] != colors.shape[0]:
        raise ValueError(f"ID count {ids.shape[0]} != color count {colors.shape[0]}")
    
    if colors.size and float(colors.max()) > 1.0:
        colors = colors / 255.0
    
    # Debug: print first few colors
    print(f"First 3 colors: {colors[:3]}")
    print(f"Last 3 colors: {colors[-3:]}")

    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    with dst_bin.open("wb") as f:
        f.write(b"CLST")  # magic
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", ids.shape[0]))  # count
        f.write(ids.astype("<u4", copy=False).tobytes())
        f.write(colors.astype("<f4", copy=False).tobytes())

def main() -> None:
    p = argparse.ArgumentParser(description="Convert clusters.pt to clusters.bin")
    p.add_argument("--model", default="clusters.pt", type=Path, help="Path to clusters.pt. (Default: clusters.pt)")
    p.add_argument("--output", default="clusters.bin", type=Path, help="Output clusters.bin path. (Default: clusters.bin)")
    args = p.parse_args()
    export_clusters_bin(args.model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
