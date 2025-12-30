#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

constant int W = 256;
constant int D = 8;
constant int SKIPS_LAYER = 4;
constant int TOTAL_INPUT_CH = 84;
constant int MULTIRES = 10;
constant int T_MULTIRES = 10;

struct CanonicalSplat {
    packed_float3 position;
    packed_half4 color;
//    packed_float4 rotation; // Somehow doesn't work, have to declare each rotation element separately.
    float rotationX;
    float rotationY;
    float rotationZ;
    float rotationW;
    packed_float3 scale;
};

inline float relu(float x) { return max(0.0, x); }

void embed_position(float3 x, float t, thread float* out_emb) {
    int idx = 0;
    out_emb[idx++] = x.x; out_emb[idx++] = x.y; out_emb[idx++] = x.z;
    for (int i = 0; i < MULTIRES; i++) {
        float freq = pow(2.0, float(i));
        float3 s = sin(x * freq);
        float3 c = cos(x * freq);
        out_emb[idx++] = s.x; out_emb[idx++] = s.y; out_emb[idx++] = s.z;
        out_emb[idx++] = c.x; out_emb[idx++] = c.y; out_emb[idx++] = c.z;
    }
    out_emb[idx++] = t;
    for (int i = 0; i < T_MULTIRES; i++) {
        float freq = pow(2.0, float(i));
        out_emb[idx++] = sin(t * freq);
        out_emb[idx++] = cos(t * freq);
    }
}

void compute_cov(float4 rot, float3 scale, thread packed_half3 &covA, thread packed_half3 &covB) {
    float x = rot.x, y = rot.y, z = rot.z, w = rot.w;
    float3x3 R = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)
    );
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    float3x3 M = R * S;
    float3x3 Sigma = M * transpose(M);
    covA = packed_half3(half(Sigma[0][0]), half(Sigma[0][1]), half(Sigma[0][2]));
    covB = packed_half3(half(Sigma[1][1]), half(Sigma[1][2]), half(Sigma[2][2]));
}

//void compute_cov(float4 rot, float3 scale, thread packed_half3 &covA, thread packed_half3 &covB) {
//    float q0 = rot.x, q1 = rot.y, q2 = rot.z, q3 = rot.w;
//    float3x3 R = float3x3(
//          2.0 * (q0 * q0 + q1 * q1) - 1,    2.0 * (q1 * q2 - q0 * q3),      2.0 * (q1 * q3 + q0 * q2),
//          2.0 * (q1 * q2 + q0 * q3),       2.0 * (q0 * q0 + q2 * q2) - 1, 2.0 * (q2 * q3 - q0 * q1),
//          2.0 * (q1 * q3 - q0 * q2),       2.0 * (q2 * q3 + q0 * q1),       2.0 * (q0 * q0 + q3 * q3) - 1
//    );
//    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
//    float3x3 M = R * S;
//    float3x3 Sigma = M * transpose(M);
//    covA = packed_half3(half(Sigma[0][0]), half(Sigma[0][1]), half(Sigma[0][2]));
//    covB = packed_half3(half(Sigma[1][1]), half(Sigma[1][2]), half(Sigma[2][2]));
//}

kernel void deformSplats(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device Splat* outSplats               [[ buffer(1) ]],
    device const float* time              [[ buffer(2) ]],
    device const float* weights           [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    // Get the id-th Gaussian in the canonical space
    CanonicalSplat input = inSplats[id];
    float3 pos = input.position;
//    float4 rot = input.rotation;
    float4 rot = float4(input.rotationX, input.rotationY, input.rotationZ, input.rotationW);
    float3 scale = input.scale;
    float t = *time;
    
    // Get embedding
    float embedded[TOTAL_INPUT_CH];
    embed_position(pos, t, embedded);
    
    float h[W];
    float h_prev[W];
    int param_offset = 0;
    
    // Define local variables to hold the network output
    float3 d_xyz = float3(0);
    float4 d_rot = float4(0); // usually w=1 identity, but d_rot is delta quaternion
    float3 d_scale = float3(0);

    // Execute MLP
    {
        device const float* w_ptr = weights + param_offset;
        device const float* b_ptr = w_ptr + (W * TOTAL_INPUT_CH);
        
        for (int row = 0; row < W; row++) {
            float sum = 0;
            for (int col = 0; col < TOTAL_INPUT_CH; col++) {
                sum += embedded[col] * w_ptr[row * TOTAL_INPUT_CH + col];
            }
            sum += b_ptr[row];
            h[row] = relu(sum);
        }
        param_offset += (W * TOTAL_INPUT_CH) + W;
    }
    
    for (int layer_idx = 0; layer_idx < D - 1; layer_idx++) {
        bool is_skip_layer = (layer_idx == SKIPS_LAYER);
        int input_dim = is_skip_layer ? (TOTAL_INPUT_CH + W) : W;
        
        for(int k=0; k<W; k++) h_prev[k] = h[k];
        
        device const float* w_ptr = weights + param_offset;
        device const float* b_ptr = w_ptr + (W * input_dim);
        
        for (int row = 0; row < W; row++) {
            float sum = 0;
            for (int col = 0; col < input_dim; col++) {
                float val;
                if (is_skip_layer) {
                     if (col < TOTAL_INPUT_CH) val = embedded[col];
                     else val = h_prev[col - TOTAL_INPUT_CH];
                } else {
                    val = h_prev[col];
                }
                sum += val * w_ptr[row * input_dim + col];
            }
            sum += b_ptr[row];
            h[row] = relu(sum);
        }
        param_offset += (W * input_dim) + W;
    }
    
    // Get d_xyz
    {
        int out_dim = 3;
        device const float* w_ptr = weights + param_offset;
        device const float* b_ptr = w_ptr + (out_dim * W);
        for (int r = 0; r < out_dim; r++) {
            float sum = 0;
            for (int c = 0; c < W; c++) sum += h[c] * w_ptr[r * W + c];
            d_xyz[r] = sum + b_ptr[r]; // Store in local var
        }
        param_offset += (out_dim * W) + out_dim;
    }
    
    // Get d_rotation
    {
        int out_dim = 4;
        device const float* w_ptr = weights + param_offset;
        device const float* b_ptr = w_ptr + (out_dim * W);
        for (int r = 0; r < out_dim; r++) {
            float sum = 0;
            for (int c = 0; c < W; c++) sum += h[c] * w_ptr[r * W + c];
            d_rot[r] = sum + b_ptr[r]; // Store in local var
        }
        param_offset += (out_dim * W) + out_dim;
    }
    
    // Get d_scaling
    {
        int out_dim = 3;
        device const float* w_ptr = weights + param_offset;
        device const float* b_ptr = w_ptr + (out_dim * W);
        for (int r = 0; r < out_dim; r++) {
            float sum = 0;
            for (int c = 0; c < W; c++) sum += h[c] * w_ptr[r * W + c];
            d_scale[r] = sum + b_ptr[r]; // Store in local var
        }
    }
    
//    d_xyz = float3(0); // TESTING W/O MLP
//    d_rot = float4(0); // TESTING W/O MLP
//    d_scale = float3(0); // TESTING W/O MLP
//    float t_val = *time; // TESTING W/O MLP

    float3 new_pos = pos + d_xyz;
    float4 new_rot = normalize(rot) + d_rot;
    float3 new_scale = exp(scale) + d_scale;
    new_scale = log(new_scale);
    
    Splat out;
    out.position = packed_float3(new_pos);
    out.color = input.color;
//    out.color = packed_half4(half(t_val), half(t_val), half(t_val), 1.0h); // R,G,B = Time // TESTING W/O MLP
    compute_cov(new_rot, new_scale, out.covA, out.covB);
    
    outSplats[id] = out;
}
