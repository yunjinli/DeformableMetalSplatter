#include "SplatProcessing.h"

vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                               constant packed_float3* clusterColors [[ buffer(BufferIndexClusterColor) ]],
                                               constant uint* clusterIDs [[ buffer(BufferIndexClusterID) ]],
                                               constant uint* selectedClusters [[ buffer(BufferIndexSelectedClusters) ]],
                                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    Splat splat = splatArray[splatID];

    return splatVertex(splat, uniforms, vertexID % 4, splatID, clusterColors, clusterIDs, selectedClusters);
}

typedef struct {
    half4 color [[color(0)]];
    int clusterID [[color(1)]];
} FragmentOut;

fragment FragmentOut singleStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                                    int currentClusterID [[color(1)]]) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    FragmentOut out;
    out.color = half4(alpha * in.color.rgb, alpha);
    
    // Only update cluster ID if this splat is significantly visible (e.g. > 50% opacity)
    // AND if it is actually contributing (alpha > 0).
    // Valid cluster IDs are >= 0.
    // If we have a valid new ID and enough alpha, we take it.
    // Otherwise keep the old one.
    if (alpha > 0.5) {
        out.clusterID = int(in.clusterID);
    } else {
        out.clusterID = currentClusterID;
    }
    return out;
}


// === Picking Shaders ===
// These render the cluster ID to a texture for click-to-select

typedef struct {
    float4 position [[position]];
    half2 relativePosition;
    uint clusterID;
} PickingFragmentIn;

vertex PickingFragmentIn pickingSplatVertexShader(uint vertexID [[vertex_id]],
                                                   uint instanceID [[instance_id]],
                                                   constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                                   constant uint* clusterIDs [[ buffer(BufferIndexClusterID) ]],
                                                   constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[0];
    PickingFragmentIn out;
    out.clusterID = 0xFFFFFFFF;  // Invalid cluster ID

    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        out.position = float4(2, 2, 0, 1);  // Off-screen
        return out;
    }

    Splat splat = splatArray[splatID];
    
    // Get cluster ID for this splat
    if (clusterIDs != nullptr) {
        out.clusterID = clusterIDs[splatID];
    }

    // Calculate position (simplified version of splatVertex)
    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;
    
    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    float2 axis1, axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 || projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds || projectedCenter.x > bounds ||
        projectedCenter.y < -bounds || projectedCenter.y > bounds) {
        out.position = float4(2, 2, 0, 1);
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[vertexID % 4];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2 * kBoundsRadius / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;
    return out;
}

fragment uint pickingSplatFragmentShader(PickingFragmentIn in [[stage_in]]) {
    // Check if this fragment is within the splat's gaussian
    half negativeMagnitudeSquared = -dot(in.relativePosition, in.relativePosition);
    if (negativeMagnitudeSquared < -kBoundsRadiusSquared) {
        discard_fragment();
    }
    return in.clusterID;
}

// === Compute shader for picking: find splat closest to clicked screen position ===
// Projects all splats to screen space and finds the one closest to click point
// Uses depth to break ties (prefers closer splats)

typedef struct {
    float2 clickPoint;      // Screen space click point (8 bytes)
    float2 screenSize;      // Screen dimensions (8 bytes)
} PickingParams;  // 16 bytes total

// Find the splat that projects closest to the click point
kernel void findNearestSplatToScreen(
    constant Splat* splatArray [[ buffer(0) ]],
    constant uint* clusterIDs [[ buffer(1) ]],
    constant Uniforms& uniforms [[ buffer(2) ]],
    constant PickingParams& params [[ buffer(3) ]],
    device atomic_uint* minScorePacked [[ buffer(4) ]],  // Upper 16 bits: score, lower 16 bits: splatID mod 65536
    device atomic_uint* resultClusterID [[ buffer(5) ]],
    uint splatID [[ thread_position_in_grid ]]
) {
    if (splatID >= uniforms.splatCount) {
        return;
    }
    
    Splat splat = splatArray[splatID];
    
    // Project splat to screen space
    float4 viewPosition = uniforms.viewMatrix * float4(splat.position, 1);
    float4 clipPosition = uniforms.projectionMatrix * viewPosition;
    
    // Skip if behind camera or outside frustum
    if (clipPosition.w <= 0 || clipPosition.z < 0) {
        return;
    }
    
    float3 ndc = clipPosition.xyz / clipPosition.w;
    
    if (ndc.x < -1 || ndc.x > 1 || ndc.y < -1 || ndc.y > 1 || ndc.z < 0 || ndc.z > 1) {
        return;
    }
    
    // Convert to screen coordinates
    float2 screenPos;
    screenPos.x = (ndc.x * 0.5 + 0.5) * params.screenSize.x;
    screenPos.y = (1.0 - (ndc.y * 0.5 + 0.5)) * params.screenSize.y;  // Flip Y
    
    // Calculate screen distance to click point
    float2 diff = screenPos - params.clickPoint;
    float screenDistSq = dot(diff, diff);
    
    // Only consider splats within 100 pixels of click
    if (screenDistSq > 10000.0) {
        return;
    }
    
    // Score combines screen distance and depth (prefer closer splats at same screen position)
    // Score = screenDist * 1000 + depth * 100 (so screen distance dominates, but depth breaks ties)
    float viewDepth = -viewPosition.z;  // Positive depth
    float score = screenDistSq + viewDepth * 10.0;  // Weight depth less than screen distance
    
    // Pack score into upper 16 bits, but we need finer granularity
    // Use full 32 bits for score comparison
    uint scoreUint = uint(score * 100.0);
    
    // Use atomic min - lowest score wins
    uint oldScore = atomic_fetch_min_explicit(minScorePacked, scoreUint, memory_order_relaxed);
    
    // If we got the new minimum, also try to store our cluster ID
    // (This is a race condition but acceptable - we just need a valid splat at/near the min)
    if (scoreUint <= oldScore) {
        atomic_store_explicit(resultClusterID, clusterIDs[splatID], memory_order_relaxed);
    }
}
