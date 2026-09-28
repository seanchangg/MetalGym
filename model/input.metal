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

//one thread per (weight or bias, chunk of rows); partial sums land in the
//float gradient arena with atomics, so the arena must be zero beforehand
struct InputBackwardParams {
    uint rows;
    uint inDim;
    uint outDim;
    uint chunkRows;
};

kernel void inputBackward(
    device const float* obs [[buffer(0)]],
    device const bfloat* dZ [[buffer(1)]],
    device atomic_float* dW [[buffer(2)]],
    device atomic_float* dBias [[buffer(3)]],
    constant InputBackwardParams& p [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint total = (p.inDim + 1) * p.outDim;
    if (tid.x >= total) return;
    uint k = tid.x / p.outDim;
    uint col = tid.x % p.outDim;
    uint row0 = tid.y * p.chunkRows;
    uint row1 = min(row0 + p.chunkRows, p.rows);
    float acc = 0.0f;
    if (k == p.inDim) {
        for (uint row = row0; row < row1; row++) acc += float(dZ[row * p.outDim + col]);
        atomic_fetch_add_explicit(&dBias[col], acc, memory_order_relaxed);
    } else {
        for (uint row = row0; row < row1; row++) acc += obs[row * p.inDim + k] * float(dZ[row * p.outDim + col]);
        atomic_fetch_add_explicit(&dW[k * p.outDim + col], acc, memory_order_relaxed);
    }
}
