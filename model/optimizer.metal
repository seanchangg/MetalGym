//flat-parameter Adam. Weights live in one bf16 arena, gradients in a float
//arena (atomic accumulators) followed by a bf16 arena; the flat master copy
//uses the same element offsets.
#include <metal_stdlib>
using namespace metal;

struct AdamParams {
    uint count;
    uint floatCount;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float beta1Pow;
    float beta2Pow;
    float maxGradNorm;
};

inline float loadGrad(device const float* gradF, device const bfloat* gradB, uint floatCount, uint i) {
    return (i < floatCount) ? gradF[i] : float(gradB[i - floatCount]);
}

kernel void gradNormSq(
    device const float* gradF [[buffer(0)]],
    device const bfloat* gradB [[buffer(1)]],
    device atomic_float* normSq [[buffer(2)]],
    constant AdamParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    float g = (tid < p.count) ? loadGrad(gradF, gradB, p.floatCount, tid) : 0.0f;
    float partial = simd_sum(g * g);
    if (lane == 0) atomic_fetch_add_explicit(normSq, partial, memory_order_relaxed);
}

kernel void adamStep(
    device float* param [[buffer(0)]],
    device const float* gradF [[buffer(1)]],
    device const bfloat* gradB [[buffer(2)]],
    device float* momentum [[buffer(3)]],
    device float* variance [[buffer(4)]],
    device bfloat* weights [[buffer(5)]],
    device const float* normSq [[buffer(6)]],
    constant AdamParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    const float norm = sqrt(normSq[0]);
    const float scale = (p.maxGradNorm > 0.0f && norm > p.maxGradNorm) ? p.maxGradNorm / (norm + 1e-6f) : 1.0f;
    float g = loadGrad(gradF, gradB, p.floatCount, tid) * scale;
    float m = p.beta1 * momentum[tid] + (1.0f - p.beta1) * g;
    float v = p.beta2 * variance[tid] + (1.0f - p.beta2) * g * g;
    momentum[tid] = m;
    variance[tid] = v;
    float mHat = m / (1.0f - p.beta1Pow);
    float vHat = v / (1.0f - p.beta2Pow);
    float x = param[tid] - p.lr * mHat / (sqrt(vHat) + p.eps);
    param[tid] = x;
    weights[tid] = bfloat(x);
}
