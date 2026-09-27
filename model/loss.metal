#include <metal_stdlib>

struct LossParams {
    metal::uint rows;
    metal::uint cols;
    metal::uint group_size;
};

kernel void lossForward(
    device const bfloat* x [[buffer(0)]],
    device metal::uint* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    device float* lse_cache [[buffer(3)]],
    constant LossParams& p [[buffer(4)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint group_id [[threadgroup_position_in_grid]],
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg [[simdgroups_per_threadgroup]]
) {
    if (group_id >= p.rows) {
        return;
    }
    threadgroup float max_scratch[32]; // one slot per SIMD-group (max 1024/32 = 32)
    threadgroup float sum_scratch[32];
    metal::uint base = group_id * p.cols;


    // ---- row max ----
    float local_max = -INFINITY;
    for (metal::uint col = tid; col < p.cols; col += p.group_size) {
        local_max = metal::max(local_max, float(x[base + col]));
    }
    local_max = metal::simd_max(local_max);
    if (lane == 0) max_scratch[sgid] = local_max;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_max = (lane < nsg) ? max_scratch[lane] : -INFINITY;
    row_max = metal::simd_max(row_max);
    float local_sum = 0.0f;
    for (metal::uint col = tid; col < p.cols; col+= p.group_size) {
        local_sum += metal::exp(float(x[base + col])-row_max);
    }
    local_sum = metal::simd_sum(local_sum);
    if (lane ==0) {sum_scratch[sgid] = local_sum;}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_sum = (lane < nsg) ? sum_scratch[lane] : 0.0f;
    row_sum = metal::simd_sum(row_sum);

    if (tid == 0) {
        float lse = row_max + metal::log(row_sum);
        lse_cache[group_id] = lse;
        out[group_id] = lse - float(x[base + y[group_id]]);
    }
}

kernel void lossBackward(
    device const bfloat* x_cache [[buffer(0)]],
    device const metal::uint* y_cache [[buffer(1)]],
    device const float* lse_cache [[buffer(2)]],
    device bfloat* dZ [[buffer(3)]],
    constant LossParams& p [[buffer(4)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint group_id [[threadgroup_position_in_grid]],
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg [[simdgroups_per_threadgroup]]
) {
    metal::uint row = group_id;
    metal::uint base = group_id * p.cols;
    for (metal::uint col = tid; col < p.cols; col += p.group_size) {
        float prob = metal::exp(float(x_cache[base+col]) - lse_cache[row]);
        float g = prob - (col == y_cache[row] ? 1.0f : 0.0f);
        dZ[base + col] = bfloat(g / float(p.rows)); //loss is averaged over rows, so the gradient is too
    }
}
