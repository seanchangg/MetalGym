//flat-parameter Adam: gather grads -> normSq -> adamStep (clips by global norm) -> scatter bf16 weights
#include <metal_stdlib>
using namespace metal;

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
    float beta1Pow;
    float beta2Pow;
    float maxGradNorm;
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

kernel void gradNormSq(
    device const float* grad [[buffer(0)]],
    device atomic_float* normSq [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    float g = (tid < count) ? grad[tid] : 0.0f;
    float partial = simd_sum(g * g);
    if (lane == 0) atomic_fetch_add_explicit(normSq, partial, memory_order_relaxed);
}

kernel void adamStep(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* momentum [[buffer(2)]],
    device float* variance [[buffer(3)]],
    constant AdamParams& p [[buffer(4)]],
    device const float* normSq [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    const float norm = sqrt(normSq[0]);
    const float scale = (p.maxGradNorm > 0.0f && norm > p.maxGradNorm) ? p.maxGradNorm / (norm + 1e-6f) : 1.0f;
    float g = grad[tid] * scale;
    float m = p.beta1 * momentum[tid] + (1.0f - p.beta1) * g;
    float v = p.beta2 * variance[tid] + (1.0f - p.beta2) * g * g;
    momentum[tid] = m;
    variance[tid] = v;
    float mHat = m / (1.0f - p.beta1Pow);
    float vHat = v / (1.0f - p.beta2Pow);
    param[tid] -= p.lr * mHat / (sqrt(vHat) + p.eps);
}
