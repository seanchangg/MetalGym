#include <metal_stdlib>

struct SoftMaxParams {
    metal::uint rows;
    metal::uint cols;
    metal::uint group_size;
};

kernel void softmaxForward(
    device const bfloat* x [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant SoftMaxParams& p [[buffer(2)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint group_id [[threadgroup_position_in_grid]],
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg [[simdgroups_per_threadgroup]]
) {
    if (group_id >= p.rows) {
        return;
    }
    threadgroup float scratch[32]; // one slot per SIMD-group (max 1024/32 = 32)

    metal::uint row = group_id;

    // ---- row max ----
    float local_max = -INFINITY;
    for (metal::uint col = tid; col < p.cols; col += p.group_size) {
        local_max = metal::max(local_max, float(x[row * p.cols + col]));
    }
    local_max = metal::simd_max(local_max);
    if (lane == 0) scratch[sgid] = local_max;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_max = (lane < nsg) ? scratch[lane] : -INFINITY;
    row_max = metal::simd_max(row_max);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ---- exp + row sum ----
    float local_sum = 0.0f;
    for (metal::uint col = tid; col < p.cols; col += p.group_size) {
        float x_val = metal::exp(float(x[row * p.cols + col]) - row_max);
        out[row * p.cols + col] = bfloat(x_val);
        local_sum += x_val;
    }
    local_sum = metal::simd_sum(local_sum);
    if (lane == 0) scratch[sgid] = local_sum;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_normed_sum = (lane < nsg) ? scratch[lane] : 0.0f;
    row_normed_sum = metal::simd_sum(row_normed_sum);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ---- normalize ----
    float inv_sum = 1.0f / row_normed_sum;
    for (metal::uint col = tid; col < p.cols; col += p.group_size) {
        out[row * p.cols + col] = bfloat(float(out[row * p.cols + col]) * inv_sum);
    }
}

kernel void softmaxBackward(
    device const bfloat* output_cache [[buffer(0)]],
    device bfloat* dZ [[buffer(1)]],
    constant SoftMaxParams& p [[buffer(2)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint gid [[threadgroup_position_in_grid]], //row idx
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg [[simdgroups_per_threadgroup]]
) {
    if (gid >= p.rows) {
        return;
    }
    metal::uint row_start = gid * p.cols;
    threadgroup float scratch[32];

    // ---- dot = sum(s_j * dZ_j) ----
    float local_dot = 0.0f;
    for (metal::uint i = tid; i < p.cols; i += p.group_size) {
        local_dot += float(output_cache[row_start + i]) * float(dZ[row_start + i]);
    }
    local_dot = metal::simd_sum(local_dot);
    if (lane == 0) scratch[sgid] = local_dot;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float dot = (lane < nsg) ? scratch[lane] : 0.0f;
    dot = metal::simd_sum(dot);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ---- dA_i = s_i * (dZ_i - dot) ----
    for (metal::uint i = tid; i < p.cols; i += p.group_size) {
        float s  = float(output_cache[row_start + i]);
        float dz = float(dZ[row_start + i]);
        dZ[row_start + i] = bfloat(s * (dz - dot));
    }
}
