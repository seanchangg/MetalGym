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

kernel void linearBackward (
    device bfloat* x [[buffer(0)]],
    device bfloat* w [[buffer(1)]],
    device bfloat* dZ [[buffer(2)]],
    device bfloat* dX [[buffer(3)]],
    device bfloat* dW [[buffer(4)]],
    constant LinearParams& p [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint z_tiles = p.N / TILE_M;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt(x, dextents<int32_t, 2>(p.K, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt(w, dextents<int32_t, 2>(p.N, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt(dZ, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dXt(dX, dextents<int32_t, 2>(p.K, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dWt(dW, dextents<int32_t, 2>(p.N, p.K));
    constexpr auto desc_dw = matmul2d_descriptor(
        N_EMBED, // m: dW output rows = embed dim (p.K here)
        TILE_M,  // n: one 64-wide column tile of the vocab
        static_cast<int>(dynamic_extent), // k: reduce over all M rows
        true, false, false, matmul2d_descriptor::mode::multiply_accumulate); //X^T @ dZ
    constexpr auto desc_dx = matmul2d_descriptor(
        TILE_M,  // m: this threadgroup's row tile
        N_EMBED, // n: dX output cols = embed dim
        static_cast<int>(dynamic_extent), // k: reduce over the vocab
        false, true, false); //dZ @ W^T
    matmul2d<desc_dw, execution_simdgroups<4>> dW_op;
    matmul2d<desc_dx, execution_simdgroups<4>> dX_op;
    const uint global_row_start = gid * 64;
    auto mdZ = dZt.slice(0, global_row_start);
    auto mdX = dXt.slice(0, global_row_start);
    dX_op.run(mdZ, Wt, mdX);
    //dW column tiles are distributed across threadgroups; each tile reduces over the FULL
    //unsliced X/dZ, so no tile is touched twice and no rows are double-counted
    for (uint z_tile = gid; z_tile < z_tiles; z_tile += p.M/TILE_M) {
        mdZ = dZt.slice(z_tile*TILE_M, 0);
        auto mdW = dWt.slice(z_tile*TILE_M, 0);
        dW_op.run(Xt, mdZ, mdW);
    }
}
