#include <metal_stdlib>

struct ReluInputParams {
    metal::uint count;
};


kernel void reluForward(
    device bfloat* x [[buffer(0)]],
    device uchar* mask[[buffer(1)]],
    constant ReluInputParams& p [[buffer(3)]],
    metal::uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) {
        return;
    }
    mask[tid] = (x[tid] > 0.0bf) ? 1 : 0;
    x[tid] *= bfloat(mask[tid]);
}

kernel void reluBackward(
    device bfloat* dZ [[buffer(0)]],
    device uchar* mask[[buffer(1)]],
    constant ReluInputParams& p [[buffer(2)]],
    metal::uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.count) {
    return;
    }
    dZ[tid] *= mask[tid];
}
