import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import scipy.ndimage
from PIL import Image

def visualize(rgb_path, cluster_path, output_path, extract_crops_folder=None):
    # Check for metadata file to determine dimensions
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
        print("Could not determine dimensions from metadata file.")
        print(f"Expected metadata file at: {meta_path}")
        return

    print(f"Dimensions: {width}x{height}")
    
    # Read RGB
    try:
        rgb_data = np.fromfile(rgb_path, dtype=np.uint8)
    except FileNotFoundError:
        print(f"File not found: {rgb_path}")
        return

    if rgb_data.size != width * height * 4:
        print(f"Error: RGB file size {rgb_data.size} bytes does not match {width}x{height}x4 pixels")
        return
    
    rgb_img = rgb_data.reshape((height, width, 4))
    # Metal saved as BGRA8Unorm
    # Convert BGRA to RGBA
    rgb_img = rgb_img[..., [2, 1, 0, 3]]
    
    # Read Clusters
    try:
        cluster_data = np.fromfile(cluster_path, dtype=np.int32)
    except FileNotFoundError:
        print(f"File not found: {cluster_path}")
        return

    if cluster_data.size != width * height:
        print(f"Error: Cluster file size {cluster_data.size*4} bytes does not match {width}x{height} pixels")
        return

    cluster_img = cluster_data.reshape((height, width))
    
    # Hole Filling
    # Replace -1 with nearest valid neighbor
    # We treat -1 as "empty".
    # Mask of invalid pixels
    invalid_mask = (cluster_img == -1)
    
    if np.any(invalid_mask):
        print("Filling holes in cluster map...")
        
        # 1. Topological Fill: Fill holes strictly enclosed by a single cluster
        # This handles large internal holes that simple dilation might miss or that we want to preserve if they weren't enclosed.
        unique_ids = np.unique(cluster_img)
        print(f"Performing topological fill on {len(unique_ids)} clusters...")
        
        for cid in unique_ids:
            if cid == -1: continue
            
            # Create binary mask for this cluster
            cluster_mask = (cluster_img == cid)
            
            # Fill holes in the binary mask
            # structure=None implies 3x3 cross (4-connectivity) or square (8)? default is 3x3 structuring element.
            filled_mask = scipy.ndimage.binary_fill_holes(cluster_mask)
            
            # Identify purely filled pixels (were 0, became 1)
            # We ONLY check where the image was originally -1 to ensure we don't overwrite other valid clusters
            holes_to_fill = filled_mask & (cluster_img == -1)
            
            if np.any(holes_to_fill):
                cluster_img[holes_to_fill] = cid
                
        # Re-evaluate invalid mask after topological fill
        invalid_mask = (cluster_img == -1)

    if np.any(invalid_mask):
        print("Performing distance-based fill for remaining gaps...")
        # 2. Distance-Based Fill: Handle boundaries and gaps
        # Iteratively replace -1 with nearest valid pixel (Voronoi regions) but limit the distance
        # to avoid filling the entire background.
        distances, indices = scipy.ndimage.distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        
        # Fill everything with nearest neighbor
        cluster_img = cluster_img[tuple(indices)]
        
        # Re-invalidate pixels that are too far from any valid cluster (e.g. background)
        # Threshold of 20 pixels (radius) fills small holes/gaps but leaves large areas empty
        cluster_img[distances > 20] = -1

    # Visualization
    unique_clusters = np.unique(cluster_img)
    print(f"Found {len(unique_clusters)} unique clusters after filling.")
    
    # Extract Crops if requested
    if extract_crops_folder:
        if not os.path.exists(extract_crops_folder):
            os.makedirs(extract_crops_folder)
            
        print(f"Extracting {len(unique_clusters)} crops to {extract_crops_folder}...")
        
        for cid in unique_clusters:
            if cid == -1: continue
            
            # Binary mask for this cluster
            mask = (cluster_img == cid)
            
            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                continue
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop the bounding box region directly from the original image
            # (keeps all pixels including other clusters that overlap)
            crop = rgb_img[rmin:rmax+1, cmin:cmax+1]
            
            # Save
            # Convert numpy array (H,W,4 or 3) to PIL Image
            # Ensure it is uint8
            crop_uint8 = crop.astype(np.uint8)
            im = Image.fromarray(crop_uint8)
            
            crop_path = os.path.join(extract_crops_folder, f"cluster_{cid}.png")
            im.save(crop_path)
            
        print("Crop extraction complete.")
    
    # Generate random colors for each cluster
    np.random.seed(42)
    cluster_colors = {}
    
    # Create overlay image
    # We will use matplotlib for blending, so we create an RGBA image for the overlay
    
    # Map IDs to colors
    # Use sparse approach since IDs can be large
    r_channel = np.zeros((height, width), dtype=np.uint8)
    g_channel = np.zeros((height, width), dtype=np.uint8)
    b_channel = np.zeros((height, width), dtype=np.uint8)
    a_channel = np.zeros((height, width), dtype=np.uint8)
    
    for cid in unique_clusters:
        if cid == -1: continue
        
        # Generate random bright color
        color = np.random.randint(50, 255, 3)
        
        mask = (cluster_img == cid)
        r_channel[mask] = color[0]
        g_channel[mask] = color[1]
        b_channel[mask] = color[2]
        a_channel[mask] = 180 # Alpha
        
    vis_img = np.stack([r_channel, g_channel, b_channel, a_channel], axis=-1)
    
    # Plotting
    plt.figure(figsize=(15, 7))
    
    # 1. Raw RGB
    plt.subplot(1, 2, 1)
    plt.title("Rendered RGB")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    # 2. Overlay
    plt.subplot(1, 2, 2)
    plt.title("Cluster ID Overlay")
    plt.imshow(rgb_img)
    plt.imshow(vis_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize captured RGB and Cluster ID frames.")
    parser.add_argument("rgb", help="Path to .raw RGB file")
    parser.add_argument("cluster", help="Path to .bin Cluster ID file")
    parser.add_argument("output", help="Path to output PNG file")
    parser.add_argument("--extract_crops", help="Folder to extract individual cluster crops to", default=None)
    args = parser.parse_args()
    
    visualize(args.rgb, args.cluster, args.output, args.extract_crops)
