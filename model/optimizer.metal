#include <metal_stdlib>
using namespace metal;

//Adam on a flat float master copy of every parameter. The host registers each
//parameter with an offset into the flat layout; gather* copies that
//parameter's gradient buffer into the flat gradient, adamStep updates the
//master copy, and scatterWeight writes the bf16 weights the kernels read.
struct FlatCopyParams {
    uint count;
    uint offset;
};

struct AdamParams {
    uint count;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float beta1Pow;  //beta1^t, for bias correction
    float beta2Pow;  //beta2^t
    float gradScale; //global-norm clipping factor, 1 when unclipped
};

kernel void gatherGradFloat(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    dst[p.offset + tid] = src[tid];
}

kernel void gatherGradBfloat(
    device const bfloat* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    dst[p.offset + tid] = float(src[tid]);
}

kernel void scatterWeight(
    device const float* master [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    dst[tid] = bfloat(master[p.offset + tid]);
}

kernel void adamStep(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* momentum [[buffer(2)]],
    device float* variance [[buffer(3)]],
    constant AdamParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    float g = grad[tid] * p.gradScale;
    float m = p.beta1 * momentum[tid] + (1.0f - p.beta1) * g;
    float v = p.beta2 * variance[tid] + (1.0f - p.beta2) * g * g;
    momentum[tid] = m;
    variance[tid] = v;
    float mHat = m / (1.0f - p.beta1Pow);
    float vHat = v / (1.0f - p.beta2Pow);
    param[tid] -= p.lr * mHat / (sqrt(vHat) + p.eps);
}
