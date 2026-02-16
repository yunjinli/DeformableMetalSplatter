#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

struct CanonicalSplat {
    packed_float3 position;
    packed_half4 color;
    float rotationX;
    float rotationY;
    float rotationZ;
    float rotationW;
    packed_float3 scale;
};

// Covariance computation from quaternion (x,y,z,w) and linear scale.
// Metal float3x3(9 scalars) fills column-major, so we transpose the standard
// quaternion-to-rotation-matrix formula to get correct column vectors.
void compute_cov(float4 rot, float3 scale, thread packed_half3 &covA, thread packed_half3 &covB) {
    float x = rot.x, y = rot.y, z = rot.z, w = rot.w;
    float3x3 R = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w),       2.0 * (x * z - y * w),
        2.0 * (x * y - z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
        2.0 * (x * z + y * w),       2.0 * (y * z - x * w),       1.0 - 2.0 * (x * x + y * y)
    );
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    float3x3 M = R * S;
    float3x3 Sigma = M * transpose(M);
    covA = packed_half3(half(Sigma[0][0]), half(Sigma[0][1]), half(Sigma[0][2]));
    covB = packed_half3(half(Sigma[1][1]), half(Sigma[1][2]), half(Sigma[2][2]));
}

// Extract xyz and t from the canonical Gaussians.
kernel void extract_graph_inputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device float* outXYZ                [[ buffer(1) ]],
    device float* outT                  [[ buffer(2) ]],
    constant float& time                [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat s = inSplats[id];
    outXYZ[id * 3 + 0] = s.position.x;
    outXYZ[id * 3 + 1] = s.position.y;
    outXYZ[id * 3 + 2] = s.position.z;
    outT[id] = time;
}

kernel void fill_time(
    device float* outT       [[ buffer(0) ]],
    constant float& time     [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    outT[id] = time;
}

// Apply d_xyz, d_rotation, d_scaling to the canonical Gaussians.
//
// NOTE: CanonicalSplat.scale is in LINEAR space (asLinearFloat already applied exp() to the
// PLY log-space scale values). The deformation network's d_scale is also a linear-space delta.
// compute_cov() expects linear-space scale for building the S diagonal matrix.
kernel void apply_graph_outputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device const float* dXYZ              [[ buffer(1) ]],
    device const float* dRot              [[ buffer(2) ]],
    device const float* dScale            [[ buffer(3) ]],
    device Splat* outSplats               [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat input = inSplats[id];
    
    float3 d_xyz = float3(dXYZ[id*3+0], dXYZ[id*3+1], dXYZ[id*3+2]);
    float3 new_pos = input.position + d_xyz;
    
    // Rotation: canonical is stored as (x, y, z, w)
    // Swizzle network rotation output (w,x,y,z) → (x,y,z,w) and add
    float4 rot = float4(input.rotationX, input.rotationY, input.rotationZ, input.rotationW);
    float4 d_rotation_raw = float4(dRot[id*4+0], dRot[id*4+1], dRot[id*4+2], dRot[id*4+3]);
    float4 d_rotation = d_rotation_raw.yzwx;
    float4 new_rot = normalize(rot) + d_rotation;
    new_rot = normalize(new_rot);
    
    // Scale: input.scale is already linear; d_scale is a linear-space delta
    float3 d_scaling = float3(dScale[id*3+0], dScale[id*3+1], dScale[id*3+2]);
    float3 new_scale = input.scale + d_scaling;
    
    Splat out;
    out.position = packed_float3(new_pos);
    out.color = input.color;
    compute_cov(new_rot, new_scale, out.covA, out.covB);
    
    outSplats[id] = out;
}

