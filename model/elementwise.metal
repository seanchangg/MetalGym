#include <metal_stdlib>
using namespace metal;

kernel void residualAdd(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    out[tid] = bfloat(float(a[tid]) + float(b[tid]));
}

//out[i] = bf16(sum over chunks of part[c * len + i])
struct ReduceParams {
    uint len;
    uint chunks;
};

kernel void reducePartials(
    device const float* part [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant ReduceParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.len) return;
    float acc = 0.0f;
    for (uint c = 0; c < p.chunks; c++) acc += part[c * p.len + tid];
    out[tid] = bfloat(acc);
}
