//GAE over T slots, one thread per env; normalizeAdvantage runs as one threadgroup
#include <metal_stdlib>
using namespace metal;

struct GaeParams {
    uint rows;
    uint horizon;
    uint headDim;
    uint actDim;
    float gamma;
    float lambda;
};

kernel void gaeBackward(
    device const float* reward [[buffer(0)]],
    device const uchar* done [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const bfloat* bootstrapHead [[buffer(3)]],
    device float* advantage [[buffer(4)]],
    device float* ret [[buffer(5)]],
    constant GaeParams& p [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.rows) return;
    float lastAdv = 0.0f;
    float nextValue = float(bootstrapHead[i * p.headDim + p.actDim]);
    for (int t = int(p.horizon) - 1; t >= 0; t--) {
        const uint k = uint(t) * p.rows + i;
        const float nonTerminal = done[k] ? 0.0f : 1.0f;
        const float delta = reward[k] + p.gamma * nextValue * nonTerminal - value[k];
        lastAdv = delta + p.gamma * p.lambda * nonTerminal * lastAdv;
        advantage[k] = lastAdv;
        ret[k] = lastAdv + value[k];
        nextValue = value[k];
    }
}

struct NormalizeParams {
    uint rows;
};

kernel void normalizeAdvantage(
    device const float* adv [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant NormalizeParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[32];

    float localSum = 0.0f;
    for (uint i = tid; i < p.rows; i += ntg) localSum += adv[i];
    localSum = simd_sum(localSum);
    if (lane == 0) scratch[sgid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = (lane < nsg) ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    const float mean = total / float(p.rows);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localVar = 0.0f;
    for (uint i = tid; i < p.rows; i += ntg) {
        float d = adv[i] - mean;
        localVar += d * d;
    }
    localVar = simd_sum(localVar);
    if (lane == 0) scratch[sgid] = localVar;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = (lane < nsg) ? scratch[lane] : 0.0f;
    var = simd_sum(var) / float(p.rows);
    const float inv = 1.0f / (sqrt(var) + 1e-8f);

    for (uint i = tid; i < p.rows; i += ntg) out[i] = (adv[i] - mean) * inv;
}
