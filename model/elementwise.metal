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
