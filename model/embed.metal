#include <metal_stdlib>

struct EmbedParams {
    metal::uint t;
    metal::uint e; //embedding dimension
    metal::uint group_size;
};

kernel void embedForward (
    device const int* input [[buffer(0)]],
    device const bfloat* embed_weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant EmbedParams& p [[buffer(3)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint gid [[threadgroup_position_in_grid]]) {
    metal::uint row_start = p.e * gid;
    metal::uint embed_start = input[gid] * p.e;
    for (int i = tid; i<p.e; i += p.group_size) {
        output[row_start + i] = embed_weights[embed_start + i];
    }
}

kernel void embedBackward (
    device const int* input_cache [[buffer(0)]],
    device bfloat* dZ [[buffer(1)]],
    device metal::atomic_float* d_weights [[buffer(2)]],
    constant EmbedParams& p [[buffer(3)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint gid [[threadgroup_position_in_grid]]) {
    metal::uint row_start = p.e * gid;
    metal::uint embed_start = input_cache[gid] * p.e;
    for (int i = tid; i<p.e; i += p.group_size) {
        metal::atomic_fetch_add_explicit(&d_weights[embed_start + i], float(dZ[row_start + i]), metal::memory_order_relaxed);
    }
}