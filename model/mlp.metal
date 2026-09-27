#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "config.h"
using namespace metal;
using namespace mpp::tensor_ops;

struct mlpParams {
    uint M;
    uint N;
    uint S;
};

constant uint TILE_M = 64;
constant uint TILE_N = 64;
constant uint N_EMBED = N_EMBED_CFG;

kernel void mlpForwardOne(
    device bfloat* x [[buffer(0)]],
    device bfloat* upproj [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device uchar* mask [[buffer(3)]],
    constant mlpParams& p [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    uint global_row_start = TILE_M * gid.x; //gid.x -> row tile, gid.y -> col tile, same as computeQKV
    uint global_col_start = TILE_N * gid.y;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt (x, dextents<int32_t, 2>{p.N, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Upt (upproj, dextents<int32_t, 2>{p.N*p.S, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt (v, dextents<int32_t, 2>{p.N*p.S, p.M});

    constexpr auto desc_up = matmul2d_descriptor(
        TILE_M,
        TILE_N,
        static_cast<int>(dynamic_extent),
        false, false, false);
    

    matmul2d<desc_up, execution_simdgroups<4>> up_op;

    auto mX = Xt.slice(0, global_row_start);
    auto mUp = Upt.slice(global_col_start, 0);
    auto mV = Vt.slice(global_col_start, global_row_start);
    up_op.run(mX, mUp, mV);
    threadgroup_barrier(mem_flags::mem_device); //matmul outputs must land before the relu pass. Lanes can cross into other rows and simdgroups don't automatically coordinate

    for (int row = sgid; row<TILE_M; row+=nsg) {
        for (int col = lane; col<TILE_N; col += 32) {
            uint idx = (global_row_start+row) *p.N*p.S + global_col_start+col;
            mask[idx] = (v[idx] > 0.0bf) ? 1 : 0;
            v[idx] *= bfloat(mask[idx]);
        }
    }
}

kernel void mlpForwardTwo (
    device bfloat* downproj [[buffer(0)]],
    device bfloat* v [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant mlpParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    uint global_row_start = TILE_M * gid.x; //gid.x -> row tile, gid.y -> col tile, same as computeQKV
    uint global_col_start = TILE_N * gid.y;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Dpt (downproj, dextents<int32_t, 2>{p.N, p.N*p.S});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt (v, dextents<int32_t, 2>{p.N*p.S, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ot (out, dextents<int32_t, 2>{p.N, p.M});
    constexpr auto desc_down = matmul2d_descriptor(
        TILE_M,
        TILE_N,
        static_cast<int>(dynamic_extent),
        false, false, false);
    matmul2d<desc_down, execution_simdgroups<4>> down_op;
    auto mV = Vt.slice(0, global_row_start);
    auto mDp = Dpt.slice(global_col_start, 0);
    auto mO = Ot.slice(global_col_start, global_row_start);
    threadgroup_barrier(mem_flags::mem_device); 
    down_op.run(mV, mDp, mO);
}

kernel void mlpBackwardOne(
    device bfloat* downproj [[buffer(1)]],
    device uchar* mask [[buffer(2)]],
    device bfloat* dZ [[buffer(3)]],
    device bfloat* dV [[buffer(5)]],
    constant mlpParams& p [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    uint global_row_start = TILE_M * gid.x; //gid.x -> row tile, gid.y -> col tile, same as computeQKV
    uint global_col_start = TILE_N * gid.y;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Dpt (downproj, dextents<int32_t, 2>{p.N, p.N*p.S});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dVt (dV, dextents<int32_t, 2>{p.N*p.S, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt (dZ, dextents<int32_t, 2>{p.N, p.M});

    constexpr auto desc_dv = matmul2d_descriptor(
        TILE_M,
        TILE_N,
        static_cast<int>(dynamic_extent),
        false, true, false); //dZ @ Dp^T
    

    matmul2d<desc_dv, execution_simdgroups<4>> dV_op;

    auto mdZ = dZt.slice(0, global_row_start);
    auto mDp = Dpt.slice(global_col_start, 0);
    auto mdV = dVt.slice(global_col_start, global_row_start);

    dV_op.run(mdZ, mDp, mdV);
    threadgroup_barrier(mem_flags::mem_device); 
    for (int row = sgid; row<TILE_M; row+=nsg) {
        for (int col = lane; col<TILE_N; col += 32) {
            uint idx = (global_row_start+row)*p.N*p.S + global_col_start+ col;
            dV[idx] *= bfloat(mask[idx]);
        }
    }
}


kernel void mlpBackwardTwo (
    device bfloat* x [[buffer(0)]],
    device bfloat* dX [[buffer(1)]],
    device bfloat* dV [[buffer(2)]],
    device bfloat* upproj [[buffer(3)]],
    device bfloat* dUp [[buffer(4)]],
    device bfloat* v [[buffer(5)]],
    device bfloat* dZ [[buffer(6)]],
    device bfloat* dDp [[buffer(7)]],
    constant mlpParams& p [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]]
) {
    //two independent tilings share this 1-D grid: dUp/dDp strips are indexed
    //by hidden column tile (N*S/64 of them), the dX output by M x N tile
    //((M/64)*(N/64) of them). The counts only coincide when S*N == M*N/64*64,
    //so each half is guarded by its own bound and the host dispatches the max.
    uint hidden_tiles = p.N * p.S / TILE_M;
    uint dx_tiles = (p.M / TILE_M) * (p.N / TILE_N);
    uint tile_row = gid % (p.M / TILE_M);
    uint tile_col = gid / (p.M / TILE_M);
    uint global_row_start = TILE_M * tile_row;
    uint global_col_start = TILE_N * tile_col;
    uint hidden_col_start = TILE_M * gid;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt (x, dextents<int32_t, 2>{p.N, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dXt (dX, dextents<int32_t, 2>{p.N, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Upt (upproj, dextents<int32_t, 2>{p.N*p.S, p.N});

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dVt (dV, dextents<int32_t, 2>{p.N*p.S, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dUppt (dUp, dextents<int32_t, 2>{p.N*p.S, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt (v, dextents<int32_t, 2>{p.N*p.S, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt (dZ, dextents<int32_t, 2>{p.N, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dDpt (dDp, dextents<int32_t, 2>{p.N, p.N*p.S});

    constexpr auto desc_dup = matmul2d_descriptor(
            N_EMBED, // X^T @ dV tile: (N x M) @ (M x 64)
            TILE_M,
            static_cast<int>(dynamic_extent),
            true, false, false);
    constexpr auto desc_ddp = matmul2d_descriptor(
            TILE_M, // V tile^T @ dZ: (64 x M) @ (M x N)
            N_EMBED,
            static_cast<int>(dynamic_extent),
            true, false, false);
    constexpr auto desc_dx = matmul2d_descriptor(
        TILE_M,
        TILE_N,
        static_cast<int>(dynamic_extent),
        false, true, false); //dV @ Up^T

    matmul2d<desc_dx, execution_simdgroups<4>> dX_op;
    matmul2d<desc_dup, execution_simdgroups<4>> dUp_op;
    matmul2d<desc_ddp, execution_simdgroups<4>> dDp_op;

    if (gid < hidden_tiles) {
        auto mdV = dVt.slice(hidden_col_start, 0);
        auto mdUp = dUppt.slice(hidden_col_start, 0);
        dUp_op.run(Xt, mdV, mdUp);

        auto mV = Vt.slice(hidden_col_start, 0);
        auto mdDp = dDpt.slice(0, hidden_col_start);
        dDp_op.run(mV, dZt, mdDp);
    }
    if (gid < dx_tiles) {
        auto mdX = dXt.slice(global_col_start, global_row_start);
        auto mdV = dVt.slice(0, global_row_start);
        auto mUp = Upt.slice(0, global_col_start);
        dX_op.run(mdV, mUp, mdX);
    }
}
