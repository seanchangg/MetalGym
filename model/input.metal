//obs projection: out = obs @ W + b; backward reduces over the batch, no dX
#include <metal_stdlib>
using namespace metal;

struct InputParams {
    uint rows;
    uint inDim;
    uint outDim;
};

kernel void inputForward(
    device const float* obs [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant InputParams& p [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    if (col >= p.outDim || row >= p.rows) return;
    float acc = float(bias[col]);
    for (uint k = 0; k < p.inDim; k++) {
        acc += obs[row * p.inDim + k] * float(w[k * p.outDim + col]);
    }
    out[row * p.outDim + col] = bfloat(acc);
}

kernel void inputBackward(
    device const float* obs [[buffer(0)]],
    device const bfloat* dZ [[buffer(1)]],
    device float* dW [[buffer(2)]],
    device float* dBias [[buffer(3)]],
    constant InputParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = (p.inDim + 1) * p.outDim;
    if (tid >= total) return;
    uint k = tid / p.outDim;
    uint col = tid % p.outDim;
    float acc = 0.0f;
    if (k == p.inDim) {
        for (uint row = 0; row < p.rows; row++) {
            acc += float(dZ[row * p.outDim + col]);
        }
        dBias[col] = acc;
    } else {
        for (uint row = 0; row < p.rows; row++) {
            acc += obs[row * p.inDim + k] * float(dZ[row * p.outDim + col]);
        }
        dW[k * p.outDim + col] = acc;
    }
}
