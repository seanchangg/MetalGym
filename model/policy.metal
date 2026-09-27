#include <metal_stdlib>
using namespace metal;

//PPO actor-critic loss. The head output row for one sample is
//  [logits(actDim) | value | zero padding up to headDim]
//so one 64-wide linear layer serves both the policy and the value function.
struct PolicyParams {
    uint rows;         //samples in the batch
    uint actDim;       //logits per row
    uint headDim;      //row stride of the head output
    float clip;        //PPO ratio clip epsilon
    float valueCoef;   //weight of the value loss
    float entropyCoef; //weight of the entropy bonus
};

//loss per sample:
//  -min(ratio * A, clamp(ratio, 1 - clip, 1 + clip) * A)
//  + valueCoef * 0.5 * (v - R)^2
//  - entropyCoef * H(pi)
//averaged over rows. Writes dZ for the head row and five stats per row:
//policy loss, value loss, entropy, approx KL, clip fraction.
kernel void policyLossBackward(
    device const bfloat* head [[buffer(0)]],
    device const uint* action [[buffer(1)]],
    device const float* advantage [[buffer(2)]],
    device const float* ret [[buffer(3)]],
    device const float* oldLogProb [[buffer(4)]],
    device bfloat* dZ [[buffer(5)]],
    device float* stats [[buffer(6)]],
    constant PolicyParams& p [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    const uint base = row * p.headDim;
    const float scale = 1.0f / float(p.rows);

    //log softmax over the logits
    float rowMax = -INFINITY;
    for (uint j = 0; j < p.actDim; j++) rowMax = max(rowMax, float(head[base + j]));
    float sum = 0.0f;
    for (uint j = 0; j < p.actDim; j++) sum += exp(float(head[base + j]) - rowMax);
    const float lse = rowMax + log(sum);

    //entropy of the current policy
    float entropy = 0.0f;
    for (uint j = 0; j < p.actDim; j++) {
        float logp = float(head[base + j]) - lse;
        entropy -= exp(logp) * logp;
    }

    //clipped surrogate
    const uint a = action[row];
    const float adv = advantage[row];
    const float logpA = float(head[base + a]) - lse;
    const float ratio = exp(logpA - oldLogProb[row]);
    const float surr1 = ratio * adv;
    const float surr2 = clamp(ratio, 1.0f - p.clip, 1.0f + p.clip) * adv;
    const float policyLoss = -min(surr1, surr2);
    //d(policyLoss)/d(logpA): zero when the clipped branch is active
    const float gLogp = (surr1 <= surr2) ? -adv * ratio : 0.0f;

    //dLoss/dlogit_j = gLogp * ([j == a] - p_j) + entropyCoef * p_j * (log p_j + H)
    for (uint j = 0; j < p.actDim; j++) {
        float logp = float(head[base + j]) - lse;
        float prob = exp(logp);
        float g = gLogp * ((j == a ? 1.0f : 0.0f) - prob)
                + p.entropyCoef * prob * (logp + entropy);
        dZ[base + j] = bfloat(g * scale);
    }

    //value head
    const float v = float(head[base + p.actDim]);
    const float vErr = v - ret[row];
    dZ[base + p.actDim] = bfloat(p.valueCoef * vErr * scale);

    //padding columns carry no gradient
    for (uint j = p.actDim + 1; j < p.headDim; j++) dZ[base + j] = bfloat(0.0f);

    stats[row * 5 + 0] = policyLoss;
    stats[row * 5 + 1] = 0.5f * p.valueCoef * vErr * vErr;
    stats[row * 5 + 2] = entropy;
    stats[row * 5 + 3] = oldLogProb[row] - logpA;
    stats[row * 5 + 4] = (fabs(ratio - 1.0f) > p.clip) ? 1.0f : 0.0f;
}
