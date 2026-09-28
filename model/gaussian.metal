//Gaussian policy head for continuous actions. Head row = [mean(ACT_DIM) |
//value | zero pad]. The log standard deviation is a separate ACT_DIM
//parameter in the weight arena, shared by every row (state independent).
#include <metal_stdlib>
using namespace metal;

struct SampleParams {
    uint rows;
    uint actDim;
    uint headDim;
};

struct PolicyParams {
    uint rows;
    uint actDim;
    uint headDim;
    float clip;
    float valueCoef;
    float entropyCoef;
};

constant float LOG_SQRT_2PI = 0.9189385332046727f; //0.5 * log(2 * pi)

inline uint gaussRngNext(thread uint& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

//Box-Muller: two uniforms give two standard normals
inline float2 gaussPair(thread uint& s) {
    const float u1 = 1.0f - float(gaussRngNext(s) >> 8) * (1.0f / 16777216.0f); //(0, 1], log stays finite
    const float u2 = float(gaussRngNext(s) >> 8) * (1.0f / 16777216.0f);
    const float r = sqrt(-2.0f * log(u1));
    const float theta = 2.0f * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

//action = mean + std * z per env row. The log-prob uses the stored action so
//the loss kernel recomputes the same value at ratio 1.
kernel void sampleGaussian(
    device const bfloat* head [[buffer(0)]],
    device uint* rng [[buffer(1)]],
    device float* action [[buffer(2)]],
    device float* logProb [[buffer(3)]],
    device float* value [[buffer(4)]],
    constant SampleParams& p [[buffer(5)]],
    device const bfloat* logStd [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    const uint base = row * p.headDim;
    const uint actBase = row * p.actDim;
    uint s = rng[row];
    float lp = 0.0f;
    float2 z = float2(0.0f);
    for (uint k = 0; k < p.actDim; k++) {
        if ((k & 1u) == 0u) z = gaussPair(s);
        const float zk = (k & 1u) ? z.y : z.x;
        const float ls = float(logStd[k]);
        const float mu = float(head[base + k]);
        const float a = mu + exp(ls) * zk;
        const float d = (a - mu) * exp(-ls);
        action[actBase + k] = a;
        lp += -0.5f * d * d - ls - LOG_SQRT_2PI;
    }
    rng[row] = s;
    logProb[row] = lp;
    value[row] = float(head[base + p.actDim]);
}

//PPO loss + gradient for one head row. dZ gets the mean and value gradients.
//The log_std gradient is a batch sum, reduced per SIMD group and added to the
//float gradient arena with atomics, so the arena must be zero beforehand.
kernel void gaussianLossBackward(
    device const bfloat* head [[buffer(0)]],
    device const float* action [[buffer(1)]],
    device const float* advantage [[buffer(2)]],
    device const float* ret [[buffer(3)]],
    device const float* oldLogProb [[buffer(4)]],
    device bfloat* dZ [[buffer(5)]],
    device float* stats [[buffer(6)]],
    constant PolicyParams& p [[buffer(7)]],
    device const bfloat* logStd [[buffer(8)]],
    device atomic_float* dLogStd [[buffer(9)]],
    uint row [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    //rows is a multiple of 64, so every SIMD group is full and the simd_sum
    //below sees no early-out lanes
    if (row >= p.rows) return;
    const uint base = row * p.headDim;
    const uint actBase = row * p.actDim;
    const float scale = 1.0f / float(p.rows);

    float logp = 0.0f;
    float entropy = 0.0f;
    for (uint k = 0; k < p.actDim; k++) {
        const float ls = float(logStd[k]);
        const float d = (action[actBase + k] - float(head[base + k])) * exp(-ls);
        logp += -0.5f * d * d - ls - LOG_SQRT_2PI;
        entropy += ls + LOG_SQRT_2PI + 0.5f;
    }

    const float adv = advantage[row];
    const float ratio = exp(logp - oldLogProb[row]);
    const float surr1 = ratio * adv;
    const float surr2 = clamp(ratio, 1.0f - p.clip, 1.0f + p.clip) * adv;
    const float policyLoss = -min(surr1, surr2);
    const float gLogp = (surr1 <= surr2) ? -adv * ratio : 0.0f;

    for (uint k = 0; k < p.actDim; k++) {
        const float ls = float(logStd[k]);
        const float invStd = exp(-ls);
        const float d = (action[actBase + k] - float(head[base + k])) * invStd;
        //d logp / d mean = d / std; d logp / d log_std = d^2 - 1; d(-H) / d log_std = -1
        dZ[base + k] = bfloat(gLogp * d * invStd * scale);
        const float gLs = (gLogp * (d * d - 1.0f) - p.entropyCoef) * scale;
        const float partial = simd_sum(gLs);
        if (lane == 0) atomic_fetch_add_explicit(&dLogStd[k], partial, memory_order_relaxed);
    }

    const float v = float(head[base + p.actDim]);
    const float vErr = v - ret[row];
    dZ[base + p.actDim] = bfloat(p.valueCoef * vErr * scale);

    for (uint j = p.actDim + 1; j < p.headDim; j++) dZ[base + j] = bfloat(0.0f);

    stats[row * 5 + 0] = policyLoss;
    stats[row * 5 + 1] = 0.5f * p.valueCoef * vErr * vErr;
    stats[row * 5 + 2] = entropy;
    stats[row * 5 + 3] = oldLogProb[row] - logp;
    stats[row * 5 + 4] = (fabs(ratio - 1.0f) > p.clip) ? 1.0f : 0.0f;
}
