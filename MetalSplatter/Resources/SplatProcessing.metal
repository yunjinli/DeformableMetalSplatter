#import "SplatProcessing.h"

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize) {
    float invViewPosZ = 1 / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    float tanHalfFovX = 1 / projectionMatrix[0][0];
    float tanHalfFovY = 1 / projectionMatrix[1][1];
    float limX = 1.3 * tanHalfFovX;
    float limY = 1.3 * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    float focalX = screenSize.x * projectionMatrix[0][0] / 2;
    float focalY = screenSize.y * projectionMatrix[1][1] / 2;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

// Helper: check if clusterID is in the selectedClusters array
bool isClusterSelected(uint clusterID, constant uint *selectedClusters, uint count) {
    for (uint i = 0; i < count && i < kMaxSelectedClusters; i++) {
        if (selectedClusters[i] == clusterID) {
            return true;
        }
    }
    return false;
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       uint splatIndex,
                       constant packed_float3 *clusterColors,
                       constant uint *clusterIDs,
                       constant uint *selectedClusters) {
    FragmentIn out;
    
    uint thisClusterID = (clusterIDs != nullptr) ? clusterIDs[splatIndex] : 0xFFFFFFFF;
    bool clusterIsSelected = false;
    
    // Check if this cluster is in the multi-selection list
    if (uniforms.selectionMode > 0 && selectedClusters != nullptr && uniforms.selectedClusterCount > 0) {
        clusterIsSelected = isClusterSelected(thisClusterID, selectedClusters, uniforms.selectedClusterCount);
    }
    
    // In confirmed mode (selectionMode == 2), cull splats not in selection
    if (uniforms.selectionMode == 2 && uniforms.selectedClusterCount > 0) {
        if (!clusterIsSelected) {
            out.position = float4(2, 2, 0, 1);  // Off-screen = culled
            out.color = half4(0);
            return out;
        }
    }
    
    // In delete/hide mode (selectionMode == 3), cull splats IN selection
    if (uniforms.selectionMode == 3 && uniforms.selectedClusterCount > 0) {
        if (clusterIsSelected) {
            out.position = float4(2, 2, 0, 1);  // Off-screen = culled
            out.color = half4(0);
            return out;
        }
    }
    
    // Filter by single selected cluster (legacy behavior) - only when not in multi-selection mode
    if (uniforms.selectionMode == 0 && uniforms.selectedClusterID >= 0 && clusterIDs != nullptr) {
        if (int(thisClusterID) != uniforms.selectedClusterID) {
            out.position = float4(2, 2, 0, 1);  // Off-screen = culled
            out.color = half4(0);
            return out;
        }
    }

    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 ||
        projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2
        * kBoundsRadius
        / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;
    out.color = splat.color;
    
    // In selection mode (selectionMode == 1), show selected clusters in red
    if (uniforms.selectionMode == 1 && clusterIsSelected) {
        out.color = half4(1.0, 0.2, 0.2, splat.color.a);  // Red tint for selected
        return out;
    }
    
    // Depth visualization takes priority
    if (uniforms.showDepthVisualization != 0) {
        // Calculate view-space depth
        float depth = -viewPosition3.z;  // Positive depth
        
        // Normalize to [0, 1] based on depth range
        float t = saturate((depth - uniforms.depthRange.x) / (uniforms.depthRange.y - uniforms.depthRange.x));
        
        // Yellow to Red colormap (near = yellow, far = red)
        float3 rgb;
        rgb.r = 1.0;                    // Red always 1
        rgb.g = 1.0 - t;                // Green fades from 1 (yellow) to 0 (red)
        rgb.b = 0.0;                    // No blue
        
        out.color = half4(half3(rgb), splat.color.a);
    } else if (uniforms.showClusterColors != 0 && clusterColors != nullptr) {
        half3 clusterColor = half3(clusterColors[splatIndex]);
        out.color = half4(clusterColor, splat.color.a);
    }

    out.clusterID = thisClusterID;
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}

