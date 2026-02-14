import torch
import numpy as np
import argparse
import sys
import os

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Please install plyfile: pip install plyfile")
    sys.exit(1)

# Import DeformNetwork from local file
try:
    from export_deform_weights import DeformNetwork
except ImportError:
    print("Could not import DeformNetwork from export_deform_weights.py")
    sys.exit(1)

def run_deformation(ply_path, model_path, t, output_path, smooth=False):
    print(f"Processing {ply_path} with model {model_path} at t={t} (smooth={smooth})")
    
    # 1. Load PLY
    plydata = PlyData.read(ply_path)
    # Assume vertex element is named 'vertex'
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
    # standard 3DGS scales are scale_0, scale_1, scale_2 (log scale)
    s0 = np.array(vertex['scale_0'])
    s1 = np.array(vertex['scale_1'])
    s2 = np.array(vertex['scale_2'])
    scales = np.stack([s0, s1, s2], axis=1).astype(np.float32)
    
    # Extract rotations
    # standard 3DGS rotations are rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z)
    r0 = np.array(vertex['rot_0'])
    r1 = np.array(vertex['rot_1'])
    r2 = np.array(vertex['rot_2'])
    r3 = np.array(vertex['rot_3'])
    # Standard 3DGS rotation convention: (w, x, y, z) = (rot_0, rot_1, rot_2, rot_3)
    rotations = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)  # (w,x,y,z)

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
    
    # 3. Process
    t_tensor = torch.full((num_points, 1), t, dtype=torch.float32, device=device)
    positions_tensor = torch.from_numpy(positions).to(device)
    
    # Split into batches to avoid OOM
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
    
    # 4. Apply Deformations
    # Position: pos + d_xyz
    new_positions = positions + d_xyz
    
    # Rotation: normalize(rot) then optionally add d_rot
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    normalized_rot = rotations / norms
    
    if smooth:
        # Smooth mode: skip rotation AND scale deltas, keep canonical values
        # This avoids noisy rotation/scale outputs for dense recordings
        print("Smooth mode: applying position deltas only")
        new_rotations = normalized_rot
        new_scales = scales  # keep original log-space scales
    else:
        # Full mode: apply rotation deltas
        new_rotations = normalized_rot + d_rot
        # Normalize result quaternion
        new_norms = np.linalg.norm(new_rotations, axis=1, keepdims=True)
        new_rotations = new_rotations / np.maximum(new_norms, 1e-9)
        # Scale: log(exp(scale) + d_scale)
        exp_scales = np.exp(scales)
        new_exp_scales = exp_scales + d_scale
        new_scales = np.log(np.maximum(new_exp_scales, 1e-6))
    
    # 5. Write Output
    print("Saving output...")
    vertex_data = plydata['vertex'].data
    
    vertex_data['x'] = new_positions[:, 0].astype(np.float32)
    vertex_data['y'] = new_positions[:, 1].astype(np.float32)
    vertex_data['z'] = new_positions[:, 2].astype(np.float32)
    
    vertex_data['scale_0'] = new_scales[:, 0].astype(np.float32)
    vertex_data['scale_1'] = new_scales[:, 1].astype(np.float32)
    vertex_data['scale_2'] = new_scales[:, 2].astype(np.float32)
    
    # Remap output rotations back to PLY convention (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    # Rotations are always in (w,x,y,z) order, matching PLY convention
    vertex_data['rot_0'] = new_rotations[:, 0].astype(np.float32)  # w
    vertex_data['rot_1'] = new_rotations[:, 1].astype(np.float32)  # x
    vertex_data['rot_2'] = new_rotations[:, 2].astype(np.float32)  # y
    vertex_data['rot_3'] = new_rotations[:, 3].astype(np.float32)  # z
    
    PlyData([PlyElement.describe(vertex_data, 'vertex')], text=False).write(output_path)
    print(f"Saved deformed PLY to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deform a PLY file using a trained DeformNetwork.")
    parser.add_argument("ply_path", help="Input PLY file")
    parser.add_argument("model_path", help="Path to deform.pth")
    parser.add_argument("t", type=float, help="Time t (0 to 1)")
    parser.add_argument("output_path", help="Output PLY file")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Smooth mode: apply only position deltas, skip rotation and scale deltas. "
                             "Reduces noise for dense recordings.")
    
    args = parser.parse_args()
    
    if not 0.0 <= args.t <= 1.0:
        print("Warning: t should typically be between 0 and 1.")
        
    run_deformation(args.ply_path, args.model_path, args.t, args.output_path, smooth=args.smooth)
