# Progress Report – Deformation Pipeline Audit (Critical Fix)

## Root Cause

- The issue was in the `compute_cov` function in `Deform.metal`, which was constructing a transposed covariance matrix.
- The code path had been ported from Python/GLM (row-major matrices) to Metal (column-major matrices).
- The `float3x3` constructor was copied verbatim, causing Metal to build the transpose of the intended rotation matrix.

  - Wrong rotation: Metal effectively constructed R^T instead of R.
  - This then propagated into the covariance construction.

- As a result, the covariance was evaluated as the wrong expression:

  - Wrong:
    - M = R^T S
    - Σ = R^T S^2 R
  - Correct:
    - Σ = R S^2 R^T

- Practically, this meant all anisotropic Gaussians were rendered with mirrored orientations, producing significant visual noise and misalignment.

## Fixes Applied

- Rotation construction in shader: Updated `Deform.metal` so that the rotation matrix is constructed with the correct column-major layout, yielding the intended R instead of R^T.
- Scaling behavior verified: Confirmed that `CanonicalSplat` uses linear scale and that the shader adds linear scale deltas, so there is no double exponentiation of scales.
- Python helper script: Updated the help text in `run_deformation.py` to clarify that smooth mode skips both rotation and scale deltas, matching the Metal implementation (it does not apply scale deltas in smooth mode).

All changes have been pushed to `main`. The app now renders deformations with correctly oriented anisotropic Gaussians.
