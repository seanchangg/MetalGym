//softmax draw per env row using the gym's xorshift state
#include <metal_stdlib>
using namespace metal;

struct SampleParams {
    uint rows;
    uint actDim;
    uint headDim;
};

inline uint sampleRngNext(thread uint& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

kernel void sampleAction(
    device const bfloat* head [[buffer(0)]],
    device uint* rng [[buffer(1)]],
    device uint* action [[buffer(2)]],
    device float* logProb [[buffer(3)]],
    device float* value [[buffer(4)]],
    constant SampleParams& p [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    const uint base = row * p.headDim;
    float rowMax = -INFINITY;
    for (uint a = 0; a < p.actDim; a++) rowMax = max(rowMax, float(head[base + a]));
    float sum = 0.0f;
    for (uint a = 0; a < p.actDim; a++) sum += exp(float(head[base + a]) - rowMax);
    const float lse = rowMax + log(sum);

    uint s = rng[row];
    const float u = float(sampleRngNext(s) >> 8) * (1.0f / 16777216.0f) * sum;
    rng[row] = s;
    uint chosen = p.actDim - 1;
    float acc = 0.0f;
    for (uint a = 0; a < p.actDim; a++) {
        acc += exp(float(head[base + a]) - rowMax);
        if (u < acc) { chosen = a; break; }
    }
    action[row] = chosen;
    logProb[row] = float(head[base + chosen]) - lse;
    value[row] = float(head[base + p.actDim]);
}
