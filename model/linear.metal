#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "config.h"
using namespace metal;
using namespace mpp::tensor_ops;

struct LinearParams {
    uint M;
    uint N;
    uint K;
};

constant uint TILE_M = 64;
constant uint N_EMBED = N_EMBED_CFG; //descriptor m/n must be compile-time; p.N (the vocab) stays runtime and is only ever a reduction/loop bound

kernel void linearForward (
    device bfloat* x [[buffer(0)]],
    device bfloat* w [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant LinearParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]
    ) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt(x, dextents<int32_t, 2>(p.K, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt(w, dextents<int32_t, 2>(p.N, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ot(out, dextents<int32_t, 2>(p.N, p.M));
    constexpr auto desc = matmul2d_descriptor(
        64,
        64,
        static_cast<int>(dynamic_extent),
        false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mX = Xt.slice(0, gid.x * 64);
    auto mW = Wt.slice(gid.y * 64, 0);
    auto mO = Ot.slice(gid.y * 64, gid.x * 64);
    op.run(mX, mW, mO);
}

//dX = dZ @ W^T, one 64-row tile per threadgroup
kernel void linearBackward (
    device bfloat* w [[buffer(1)]],
    device bfloat* dZ [[buffer(2)]],
    device bfloat* dX [[buffer(3)]],
    constant LinearParams& p [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]]
) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt(w, dextents<int32_t, 2>(p.N, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt(dZ, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dXt(dX, dextents<int32_t, 2>(p.K, p.M));
    constexpr auto desc_dx = matmul2d_descriptor(
        TILE_M,
        N_EMBED,
        static_cast<int>(dynamic_extent),
        false, true, false); //dZ @ W^T
    matmul2d<desc_dx, execution_simdgroups<4>> dX_op;
    const uint global_row_start = gid * TILE_M;
    auto mdZ = dZt.slice(0, global_row_start);
    auto mdX = dXt.slice(0, global_row_start);
    dX_op.run(mdZ, Wt, mdX);
}

//dWPart[chunk] = X_chunk^T @ dZ_chunk (K x N float). gid.x = 64-wide column
//tile of N, gid.y = chunk. reducePartials sums the chunks.
struct LinearWeightParams {
    uint M;
    uint N;
    uint K;
    uint chunkRows;
};

kernel void linearBackwardDw (
    device bfloat* x [[buffer(0)]],
    device bfloat* dZ [[buffer(1)]],
    device float* dWPart [[buffer(2)]],
    constant LinearWeightParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint tile = gid.x;
    const uint chunk = gid.y;
    const uint row0 = chunk * p.chunkRows;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xc(x + row0 * p.K, dextents<int32_t, 2>(p.K, p.chunkRows));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZc(dZ + row0 * p.N, dextents<int32_t, 2>(p.N, p.chunkRows));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> dWP(dWPart + chunk * p.K * p.N, dextents<int32_t, 2>(p.N, p.K));
    constexpr auto desc_dw = matmul2d_descriptor(
        N_EMBED,
        TILE_M,
        static_cast<int>(dynamic_extent),
        true, false, false); //X^T @ dZ
    matmul2d<desc_dw, execution_simdgroups<4>> dW_op;
    auto mdZ = dZc.slice(tile * TILE_M, 0);
    auto mdW = dWP.slice(tile * TILE_M, 0);
    dW_op.run(Xc, mdZ, mdW);
}
