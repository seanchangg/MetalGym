#include <metal_stdlib>

struct LayernormParams {
    metal::uint rows;
    metal::uint cols;
    metal::uint group_size;
    float eps;
};

kernel void layernormForward(
    device const bfloat* x      [[buffer(0)]],
    device bfloat* x_norm       [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    device const bfloat* gamma  [[buffer(3)]],
    device const bfloat* beta   [[buffer(4)]],
    device float* stdev         [[buffer(5)]],
    constant LayernormParams& p [[buffer(6)]],
    metal::uint tid  [[thread_position_in_threadgroup]],
    metal::uint gid  [[threadgroup_position_in_grid]],
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg  [[simdgroups_per_threadgroup]],
    metal::uint ntg  [[threads_per_threadgroup]]
) {
    if (gid >= p.rows) {
        return;
    }
    const metal::uint row_start = gid * p.cols;
    threadgroup float sum_scratch[32];  // fp32 reduction scratch

    float local_sum = 0.0f;
    for (metal::uint col = tid; col < p.cols; col += ntg) {
        local_sum += float(x[row_start + col]);
    }
    local_sum = metal::simd_sum(local_sum);
    if (lane == 0) {
        sum_scratch[sgid] = local_sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_sum = (lane < nsg) ? sum_scratch[lane] : 0.0f;
    row_sum = metal::simd_sum(row_sum);
    float row_mean = row_sum / float(p.cols);

    float local_var = 0.0f;
    for (metal::uint col = tid; col < p.cols; col += ntg) {
        float d = float(x[row_start + col]) - row_mean;
        local_var += d * d;
    }
    local_var = metal::simd_sum(local_var);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (lane == 0) {
        sum_scratch[sgid] = local_var;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float row_var = (lane < nsg) ? sum_scratch[lane] : 0.0f;
    row_var = metal::simd_sum(row_var);
    row_var /= float(p.cols);

    float row_std  = metal::sqrt(row_var + p.eps);
    float row_rstd = 1.0f / row_std;

    if (tid == 0) {
        stdev[gid] = row_std;
    }

    for (metal::uint col = tid; col < p.cols; col += ntg) {
        float xn = (float(x[row_start + col]) - row_mean) * row_rstd;
        x_norm[row_start + col] = bfloat(xn);
        out[row_start + col]    = bfloat(float(gamma[col]) * xn + float(beta[col]));
    }
}


kernel void layernormBackward(
    device bfloat* dZ [[buffer(0)]],
    device const bfloat* x_norm_cache [[buffer(1)]],
    device const bfloat* gamma [[buffer(2)]],
    device const float* stdev [[buffer(3)]],
    device bfloat* dx_norm [[buffer(5)]],
    device metal::atomic_float* dgamma [[buffer(6)]],
    device metal::atomic_float* dbeta [[buffer(7)]],
    constant LayernormParams& p [[buffer(8)]],
    metal::uint tid  [[thread_position_in_threadgroup]],
    metal::uint gid  [[threadgroup_position_in_grid]],
    metal::uint lane [[thread_index_in_simdgroup]],
    metal::uint sgid [[simdgroup_index_in_threadgroup]],
    metal::uint nsg  [[simdgroups_per_threadgroup]],
    metal::uint ntg  [[threads_per_threadgroup]]
) {
    if (gid >= p.rows) {
        return;
    }
    const metal::uint row_start = gid * p.cols;
    threadgroup float sum1_scratch[32];
    threadgroup float sum2_scratch[32];

    for (int i = tid; i<p.cols; i+= p.group_size) {
        float g = float(x_norm_cache[row_start + i] * dZ[row_start + i]);
        metal::atomic_fetch_add_explicit(&dgamma[i], g, metal::memory_order_relaxed);
        metal::atomic_fetch_add_explicit(&dbeta[i], float(dZ[row_start+i]), metal::memory_order_relaxed);
        dx_norm[row_start + i] = dZ[row_start + i] * gamma[i];
    }

    float sum1 = 0.0f;
    float sum2 =0.0f;
    for (int i = tid; i<p.cols;i += p.group_size) {
        sum1 += dx_norm[row_start + i];
        sum2 += dx_norm[row_start+i] * x_norm_cache[row_start + i];
    }
    sum1 = metal::simd_sum(sum1);
    sum2 = metal::simd_sum(sum2);
    if (lane == 0) {
        sum1_scratch[sgid] = sum1;
        sum2_scratch[sgid] = sum2;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    sum1 = (lane < nsg) ? sum1_scratch[lane] : 0.0f;
    sum2 = (lane < nsg) ? sum2_scratch[lane] : 0.0f;
    sum1 = metal::simd_sum(sum1);
    sum2 = metal::simd_sum(sum2);
    float repsum1 = sum1/p.cols;
    float repsum2 = sum2/p.cols;
    float repdev = 1.0/stdev[gid];

    for (int i = tid; i<p.cols;i += p.group_size) {
        dZ[row_start + i] = bfloat(repdev * (dx_norm[row_start+i] - repsum1 - x_norm_cache[row_start +i] * repsum2));
    }
}