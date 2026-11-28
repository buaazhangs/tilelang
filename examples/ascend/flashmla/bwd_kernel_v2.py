"""v2 multi-buffered, pipelined DeepSeek V4.1 Sparse FlashMLA backward kernels for Ascend.

Improvements over v1 (bwd_kernel.py):
  - Double-buffered T.Pipelined preprocess & postprocess.
  - Main kernel L0 multi-buffering (aq:2, bq:4, cdkv:2) + hoisted p_l1/ds_l1 transpose loads.
  - Inner d-loops pipelined (T.Pipelined(4, num_stages=2)) with unit_flag_ctrl and mmad_direction.
  - Outer KV tile loop pipelined (T.Pipelined(nb, num_stages=2)) with L1/UB multi-buffering.
  - Paired 2-burst gather2 DMA (16 MTE2 requests/AIV instead of 32) with matching pair scatter.
"""

import tilelang.ascend.language as T
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
)


def sparse_attention_bwd_preprocess_v2(nq, *, sink=True, core_count=36):
    """Pipelined double-buffered Delta, LSE_tot, and dSink computation."""
    if nq <= 0 or core_count <= 0:
        raise ValueError("nq and core_count must be positive integers")
    num_aiv = min(nq, core_count * 2)

    @T.prim_func
    def main(
        O: T.Tensor((nq, 64, 512), "bfloat16"),
        dO: T.Tensor((nq, 64, 512), "bfloat16"),
        LSE: T.Tensor((nq, 64), "float32"),
        Sink: T.Tensor((64,), "float32"),
        Delta: T.Tensor((nq, 64), "float32"),
        LSE_tot: T.Tensor((nq, 64), "float32"),
        dSink: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(num_aiv) as bx:
            o_ub = T.alloc_shared((32, 512), "bfloat16")
            do_ub = T.alloc_shared((32, 512), "bfloat16")
            lse_ub = T.alloc_shared((64,), "float32")
            delta_ub = T.alloc_shared((64,), "float32")
            lse_tot_ub = T.alloc_shared((64,), "float32")
            sink_ub = T.alloc_shared((64,), "float32")
            dsink_ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions(
                {o_ub: 2, do_ub: 2, lse_ub: 2, delta_ub: 2, lse_tot_ub: 2, sink_ub: 1, dsink_ub: 1}
            )

            if sink:
                T.copy(Sink[:], sink_ub[:])
                with T.SimdVF():
                    T.fill(dsink_ub, 0.0)

            for job in T.Pipelined(T.ceildiv(nq - bx, num_aiv), num_stages=2):
                qidx = bx + job * num_aiv
                T.copy(LSE[qidx, :], lse_ub[:])
                for half in T.Pipelined(2, num_stages=2):
                    T.copy(O[qidx, half * 32 : (half + 1) * 32, :], o_ub)
                    T.copy(dO[qidx, half * 32 : (half + 1) * 32, :], do_ub)
                    with T.SimdVF():
                        mask32 = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        for r in range(32):
                            acc = T.alloc_var("float32x64", init=T.simd.vdup(0.0, "float32"))
                            for c in T.unroll(4):
                                o_v = T.simd.vld(o_ub[r, c * 128], "NORM")
                                do_v = T.simd.vld(do_ub[r, c * 128], "NORM")
                                o0 = T.simd.vcvt(o_v, "float32", part=0)
                                o1 = T.simd.vcvt(o_v, "float32", part=1)
                                do0 = T.simd.vcvt(do_v, "float32", part=0)
                                do1 = T.simd.vcvt(do_v, "float32", part=1)
                                prod = T.simd.vadd(T.simd.vmul(o0, do0, mask32), T.simd.vmul(o1, do1, mask32), mask32)
                                acc = T.simd.vadd(acc, prod, mask32)
                            row_sum = T.simd.vcadd(acc, mask32)
                            T.simd.vsts(delta_ub[half * 32 + r], row_sum, one, "ONEPT_B32")

                with T.SimdVF():
                    mask32 = T.simd.pset(32)
                    zero = T.simd.vdup(0.0, "float32")
                    l_v = T.simd.vld(lse_ub[0])
                    d_v = T.simd.vld(delta_ub[0])
                    if sink:
                        s_v = T.simd.vld(sink_ub[0])
                        ds_v = T.simd.vld(dsink_ub[0])
                        l_inv = T.simd.vcmps(l_v, T.float32(1e30), mask32, op="gt")
                        s_hi = T.simd.vcmps(s_v, T.float32(1e30), mask32, op="gt")
                        s_lo = T.simd.vcmps(s_v, T.float32(-1e30), mask32, op="lt")
                        l_safe = T.simd.vsel(zero, l_v, l_inv)
                        s_safe = T.simd.vsel(zero, s_v, s_hi)
                        s_safe = T.simd.vsel(zero, s_safe, s_lo)
                        m = T.simd.vmax(l_safe, s_safe, mask32)
                        ex = T.simd.vadd(T.simd.vexpdif(l_safe, m, mask32), T.simd.vexpdif(s_safe, m, mask32), mask32)
                        lt_fin = T.simd.vadd(m, T.simd.vln(ex, mask32), mask32)
                        contrib = T.simd.vmul(T.simd.vexpdif(s_safe, lt_fin, mask32), d_v, mask32)
                        lt = T.simd.vsel(l_v, lt_fin, s_lo)
                        lt = T.simd.vsel(s_v, lt, s_hi)
                        lt = T.simd.vsel(l_v, lt, l_inv)
                        contrib = T.simd.vsel(zero, contrib, s_lo)
                        contrib = T.simd.vsel(zero, contrib, s_hi)
                        contrib = T.simd.vsel(zero, contrib, l_inv)
                        T.simd.vsts(lse_tot_ub[0], lt, mask32)
                        T.simd.vsts(dsink_ub[0], T.simd.vsub(ds_v, contrib, mask32), mask32)
                    else:
                        T.simd.vsts(lse_tot_ub[0], l_v, mask32)

                T.copy(delta_ub[:], Delta[qidx, :])
                T.copy(lse_tot_ub[:], LSE_tot[qidx, :])

            if sink:
                T.set_atomic("add", "float32")
                T.copy(dsink_ub[:], dSink[:])
                T.set_atomic_none()

    return main


def sparse_attention_bwd_postprocess_v2(nk, *, core_count=36):
    """Pipelined double-buffered FP32 dKV_accum -> BF16 dKV conversion."""
    if nk <= 0 or core_count <= 0:
        raise ValueError("nk and core_count must be positive integers")
    total_tiles = (nk + 31) // 32
    num_aiv = min(total_tiles, core_count * 2)

    @T.prim_func
    def main(
        dKV_accum: T.Tensor((nk, 512), "float32"),
        dKV: T.Tensor((nk, 512), "bfloat16"),
    ):
        with T.Kernel(num_aiv) as bx:
            in_ub = T.alloc_shared((32, 512), "float32")
            out_ub = T.alloc_shared((32, 512), "bfloat16")
            T.annotate_buffer_versions({in_ub: 2, out_ub: 2})
            for job in T.Pipelined(T.ceildiv(total_tiles - bx, num_aiv), num_stages=2):
                tile_id = bx + job * num_aiv
                row_start = tile_id * 32
                rows = T.min(32, nk - row_start)
                T.copy(dKV_accum[row_start : row_start + rows, :], in_ub[:rows, :])
                with T.SimdVF():
                    mask16 = T.simd.pset(16)
                    mask32 = T.simd.pset(32)
                    for r in range(16):
                        for d in T.unroll(4):
                            packed = T.alloc_local((2, 2), "bfloat16x128")
                            for rr in T.unroll(2):
                                for cc in T.unroll(2):
                                    x = T.simd.vld(in_ub[r * 2 + rr, d * 128 + cc * 64])
                                    if rr == 0:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=0)
                                    else:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=1)
                            m0 = T.simd.vadd(packed[0, 0], packed[1, 0], mask16)
                            m1 = T.simd.vadd(packed[0, 1], packed[1, 1], mask16)
                            row0, row1 = T.simd.vdintlv(m0, m1)
                            T.simd.vsts(out_ub[r * 2, d * 128], row0, mask16)
                            T.simd.vsts(out_ub[r * 2 + 1, d * 128], row1, mask16)
                T.copy(out_ub[:rows, :], dKV[row_start : row_start + rows, :])

    return main


def sparse_attention_bwd_main_v2(
    nq,
    nk,
    topk,
    *,
    scale=512 ** (-0.5),
    q_strides=None,
    index_stride=None,
    core_count=36,
):
    """v2 multi-buffered, pipelined FlashMLA backward main mixed kernel (1C2V)."""
    if nq <= 0 or nk <= 0 or topk <= 0 or topk % 64:
        raise ValueError("nq/nk must be positive and topk a positive multiple of 64")
    if core_count <= 0:
        raise ValueError("core_count must be a positive integer")

    q_strides = q_strides or (64 * 512, 512, 1)
    index_stride = index_stride or topk
    nb = topk // 64
    num_blocks = min(nq, core_count)
    valid_limit = T.uint32(nk)

    @T.prim_func
    def main(
        Q: T.StridedTensor((nq, 64, 512), q_strides, "bfloat16"),
        KV: T.Tensor((nk, 512), "bfloat16"),
        dO: T.Tensor((nq, 64, 512), "bfloat16"),
        Indices: T.StridedTensor((nq, topk), (index_stride, 1), "int32"),
        LSE_tot: T.Tensor((nq, 64), "float32"),
        Delta: T.Tensor((nq, 64), "float32"),
        dQ: T.Tensor((nq, 64, 512), "bfloat16"),
        dKV_accum: T.Tensor((nk, 512), "float32"),
    ):
        with T.MixedKernel(num_blocks) as (bx, sid):
            # L1 buffers (64+64 + 64+64 + 16+16 = 288 KiB < 512 KiB)
            q_l1 = T.alloc_l1((64, 512), "bfloat16")
            do_l1 = T.alloc_l1((64, 512), "bfloat16")
            kv_l1_0 = T.alloc_l1((32, 512), "bfloat16")
            kv_l1_1 = T.alloc_l1((32, 512), "bfloat16")
            p_l1 = T.alloc_l1((64, 64), "bfloat16")
            ds_l1 = T.alloc_l1((64, 64), "bfloat16")

            # L0A (32+8+8 = 48 KiB <= 64 KiB) & L0B (64 KiB shared ring)
            aq = T.alloc_l0a((64, 128), "bfloat16")
            ap = T.alloc_l0a((64, 64), "bfloat16")
            ads = T.alloc_l0a((64, 64), "bfloat16")
            bq = T.alloc_l0b((64, 128), "bfloat16")
            bv = T.view(bq, shape=(64, 128))

            # L0C (128+16+16+64 = 224 KiB <= 256 KiB)
            cdq = T.alloc_l0c((4, 64, 128), "float32")
            cs = T.alloc_l0c((64, 64), "float32")
            cdp = T.alloc_l0c((64, 64), "float32")
            cdkv = T.alloc_l0c((64, 128), "float32")

            # UB per AIV (~220 KiB < 256 KiB; dq_final reuses dkv_full)
            kv_ub = T.alloc_shared((32, 512), "bfloat16")
            nz_ub = T.alloc_shared((33, 512), "bfloat16")
            s_ub = T.alloc_shared((32, 64), "float32")
            dp_ub = T.alloc_shared((32, 64), "float32")
            p_ub = T.alloc_shared((32, 64), "bfloat16")
            ds_ub = T.alloc_shared((32, 64), "bfloat16")
            chunk_fp32 = T.alloc_shared((32, 128), "float32")
            dkv_full = T.alloc_shared((32, 512), "float32")
            dq_final = T.view(dkv_full, shape=(32, 1024), dtype="bfloat16")
            lse_ub = T.alloc_shared((32,), "float32")
            delta_ub = T.alloc_shared((32,), "float32")
            idx_ub = T.alloc_shared((64,), "int32")
            smaller = T.alloc_shared((32,), "uint32")
            larger = T.alloc_shared((32,), "uint32")

            T.annotate_layout(
                {
                    q_l1: make_ascend_major_k_layout(q_l1),
                    do_l1: make_ascend_major_k_layout(do_l1),
                    kv_l1_0: make_ascend_major_k_layout(kv_l1_0),
                    kv_l1_1: make_ascend_major_k_layout(kv_l1_1),
                    p_l1: make_ascend_major_k_layout(p_l1),
                    ds_l1: make_ascend_major_k_layout(ds_l1),
                    aq: make_ascend_major_k_layout(aq),
                    ap: make_ascend_major_k_layout(ap),
                    ads: make_ascend_major_k_layout(ads),
                    bq: make_ascend_major_k_layout(bq),
                    bv: make_ascend_major_mn_layout(bv),
                    nz_ub: make_ascend_compact_nz_layout(nz_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    do_l1: 1,
                    kv_l1_0: 2,
                    kv_l1_1: 2,
                    p_l1: 2,
                    ds_l1: 2,
                    aq: 2,
                    ap: 1,
                    ads: 1,
                    bq: 4,
                    cs: 1,
                    cdp: 1,
                    cdq: 1,
                    cdkv: 2,
                    kv_ub: 2,
                    nz_ub: 1,
                    s_ub: 1,
                    dp_ub: 1,
                    p_ub: 1,
                    ds_ub: 1,
                    chunk_fp32: 2,
                    dkv_full: 1,
                    dq_final: 1,
                    lse_ub: 1,
                    delta_ub: 1,
                    idx_ub: 1,
                    smaller: 2,
                    larger: 2,
                }
            )
            # Zero every kv_ub version once so zero-burst pair gathers stay finite.
            with T.SimdVF():
                T.fill(kv_ub, 0)

            for job in T.Serial(T.ceildiv(nq - bx, num_blocks)):
                qidx = bx + job * num_blocks

                T.copy(Q[qidx, :, :], q_l1)
                T.copy(dO[qidx, :, :], do_l1)
                T.copy(LSE_tot[qidx, sid * 32 : sid * 32 + 32], lse_ub)
                T.copy(Delta[qidx, sid * 32 : sid * 32 + 32], delta_ub)

                for k in T.Pipelined(nb, num_stages=2):
                    # 1. Load 64 indices and pair-sort into smaller/larger for gather2.
                    T.copy(Indices[qidx, k * 64 : (k + 1) * 64], idx_ub)
                    with T.SimdVF():
                        pairs32 = T.simd.pset(32, "PAT_VL32")
                        x = T.reinterpret("uint32x64", T.simd.vld(idx_ub[0]))
                        even, odd = T.simd.vdintlv(x, x)
                        lo = T.simd.vmins(T.simd.vmin(even, odd), valid_limit)
                        hi = T.simd.vmax(even, odd)
                        T.simd.vsts(smaller[0], lo, pairs32, extent=32)
                        T.simd.vsts(larger[0], hi, pairs32, extent=32)

                    # 2. Paired 2-burst gather2 DMA (16 requests/AIV) + SIMD ND->NZ -> L1.
                    for i in T.unroll(16):
                        lo = smaller[sid * 16 + i]
                        hi = larger[sid * 16 + i]
                        second = T.if_then_else(hi < valid_limit, hi, lo)
                        n_burst = T.alloc_var("int32", init=T.if_then_else(lo < valid_limit, 2, 0))
                        safe_lo = T.if_then_else(n_burst == 0, T.uint32(0), lo)
                        stride = T.alloc_var("int64", init=T.Cast("int64", second - lo))
                        main_pairs = T.StridedTensor(
                            (nk, 2, 512),
                            (512, stride * 512, 1),
                            "bfloat16",
                            data=KV.data,
                        )
                        for _ in T.Serial(1):
                            T.assume(n_burst > 0)
                            T.assume(n_burst == 0 or safe_lo < main_pairs.shape[0])
                            T.copy(main_pairs[safe_lo, :n_burst, :], kv_ub[i * 2 : i * 2 + n_burst, :])

                    with T.SimdVF():
                        nd_mask = T.simd.pset(16)
                        src_ptr = T.simd.make_ubuf_ptr(T.access_ptr(kv_ub[0, 0], "r", extent=16384), "bfloat16")
                        dst0 = T.simd.make_ubuf_ptr(T.access_ptr(nz_ub[0, 0], "w", extent=16896), "bfloat16")
                        dst1 = T.simd.make_ubuf_ptr(T.access_ptr(nz_ub[0, 128], "w", extent=12672), "bfloat16")
                        for _col in T.unroll(2, unroll_factor=1):
                            for row in T.unroll(32):
                                x0, src_ptr = T.simd.vld(src_ptr, post_inc=128)
                                x1, src_ptr = T.simd.vld(src_ptr, post_inc=T.if_then_else(row == 31, -15744, 384))
                                config = (33 << 16) | T.if_then_else(row == 31, 497, 1)
                                dst0 = T.simd.vsstb(x0, dst0, config, nd_mask, update=True)
                                dst1 = T.simd.vsstb(x1, dst1, config, nd_mask, update=True)

                    if sid == 0:
                        T.copy(nz_ub[:32, :], kv_l1_0)
                    else:
                        T.copy(nz_ub[:32, :], kv_l1_1)

                    # 3. GEMM 1: S = Q @ KV^T -> cs (64, 64) FP32 -> s_ub (32, 64) ND per AIV.
                    T.set_mmad_direction("m")
                    for d in T.Pipelined(4, num_stages=2):
                        T.copy(q_l1[:, d * 128 : (d + 1) * 128], aq)
                        with T.Task():
                            T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bq[:32, :], l0_pitch=4)
                            T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bq[32:, :], l0_pitch=4)
                        T.gemm(
                            aq,
                            bq,
                            cs,
                            transpose_B=True,
                            clear_accum=d == 0,
                            unit_flag_ctrl=T.if_then_else(d == 3, 3, 2),
                        )
                    T.dual_copy(cs, s_ub, unit_flag_ctrl=3)

                    # 4. GEMM 2: dP = dO @ KV^T -> cdp (64, 64) FP32 -> dp_ub (32, 64) ND per AIV.
                    for d in T.Pipelined(4, num_stages=2):
                        T.copy(do_l1[:, d * 128 : (d + 1) * 128], aq)
                        with T.Task():
                            T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bq[:32, :], l0_pitch=4)
                            T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bq[32:, :], l0_pitch=4)
                        T.gemm(
                            aq,
                            bq,
                            cdp,
                            transpose_B=True,
                            clear_accum=d == 0,
                            unit_flag_ctrl=T.if_then_else(d == 3, 3, 2),
                        )
                    T.dual_copy(cdp, dp_ub, unit_flag_ctrl=3)

                    # 5. Vector: compute P and dS_scaled using pair-interleaved (smaller, larger) mask.
                    with T.SimdVF():
                        mask32 = T.simd.pset(32)
                        mask16 = T.simd.pset(16)
                        mask16_64 = T.simd.pset(16, "PAT_VL64")
                        lo_v = T.simd.vld(smaller[0])
                        hi_v = T.simd.vld(larger[0])
                        paired, _ignored = T.simd.vintlv(lo_v, hi_v)
                        col_valid = T.simd.vcmps(paired, valid_limit, mask32, op="lt")
                        for r in range(16):
                            p_pair = T.alloc_local((2,), "bfloat16x128")
                            ds_pair = T.alloc_local((2,), "bfloat16x128")
                            for rr in T.unroll(2):
                                l_brc = T.simd.vld(lse_ub[r * 2 + rr], "BRC_B32")
                                d_brc = T.simd.vld(delta_ub[r * 2 + rr], "BRC_B32")
                                row_valid = T.simd.vcmps(l_brc, T.float32(1e30), mask32, op="lt")
                                vmask = T.simd.pand(col_valid, row_valid, mask32)
                                s_row = T.simd.vld(s_ub[r * 2 + rr, 0])
                                dp_row = T.simd.vld(dp_ub[r * 2 + rr, 0])
                                p_row = T.simd.vexpdif(T.simd.vmuls(s_row, T.float32(scale), mask32), l_brc, vmask)
                                ds_row = T.simd.vmuls(
                                    T.simd.vmul(p_row, T.simd.vsub(dp_row, d_brc, mask32), vmask),
                                    T.float32(scale),
                                    vmask,
                                )
                                if rr == 0:
                                    p_pair[rr] = T.simd.vcvt(p_row, "bfloat16", mask32, part=0)
                                    ds_pair[rr] = T.simd.vcvt(ds_row, "bfloat16", mask32, part=0)
                                else:
                                    p_pair[rr] = T.simd.vcvt(p_row, "bfloat16", mask32, part=1)
                                    ds_pair[rr] = T.simd.vcvt(ds_row, "bfloat16", mask32, part=1)
                            p_merged = T.simd.vadd(p_pair[0], p_pair[1], mask16)
                            p_r0, p_r1 = T.simd.vdintlv(p_merged, p_merged)
                            T.simd.vsts(p_ub[r * 2, 0], p_r0, mask16_64, "NORM_B16", extent=64)
                            T.simd.vsts(p_ub[r * 2 + 1, 0], p_r1, mask16_64, "NORM_B16", extent=64)

                            ds_merged = T.simd.vadd(ds_pair[0], ds_pair[1], mask16)
                            ds_r0, ds_r1 = T.simd.vdintlv(ds_merged, ds_merged)
                            T.simd.vsts(ds_ub[r * 2, 0], ds_r0, mask16_64, "NORM_B16", extent=64)
                            T.simd.vsts(ds_ub[r * 2 + 1, 0], ds_r1, mask16_64, "NORM_B16", extent=64)

                    T.dual_copy(p_ub, p_l1)
                    T.dual_copy(ds_ub, ds_l1)

                    # 6. GEMM 3: dQ += dS_scaled @ KV -> cdq (4, 64, 128) FP32 resident in L0C.
                    T.set_mmad_direction("n")
                    T.copy(ds_l1, ads)
                    for d in T.Pipelined(4, num_stages=2):
                        with T.Task():
                            T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bv[:32, :], l0_pitch=8)
                            T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bv[32:, :], l0_pitch=8)
                        T.gemm(ads, bv, cdq[d, :, :], clear_accum=k == 0)

                    # 7. GEMM 4 + 5: hoist p_l1^T and ds_l1^T once; pipeline d across double-buffered cdkv.
                    T.copy(p_l1, ap, transpose=True)
                    T.copy(ds_l1, ads, transpose=True)
                    for d in T.Pipelined(4, num_stages=2):
                        T.copy(do_l1[:, d * 128 : (d + 1) * 128], bv)
                        T.gemm(ap, bv, cdkv, clear_accum=True, unit_flag_ctrl=2)

                        T.copy(q_l1[:, d * 128 : (d + 1) * 128], bv)
                        T.gemm(ads, bv, cdkv, clear_accum=False, unit_flag_ctrl=3)

                        T.dual_copy(cdkv, chunk_fp32, unit_flag_ctrl=3)
                        with T.SimdVF():
                            mask32 = T.simd.pset(32)
                            for r in range(32):
                                T.simd.vsts(dkv_full[r, d * 128], T.simd.vld(chunk_fp32[r, 0]), mask32)
                                T.simd.vsts(dkv_full[r, d * 128 + 64], T.simd.vld(chunk_fp32[r, 64]), mask32)

                    # 8. Paired 2-burst FP32 atomic scatter-add (scatter2: 0/1/2 bursts, branchless for AutoSchedule).
                    T.set_atomic("add", "float32")
                    for i in T.unroll(16):
                        lo = smaller[sid * 16 + i]
                        hi = larger[sid * 16 + i]
                        n_burst = T.alloc_var(
                            "int32",
                            init=T.if_then_else(hi < valid_limit, 2, T.if_then_else(lo < valid_limit, 1, 0)),
                        )
                        safe_lo = T.if_then_else(n_burst == 0, T.uint32(0), lo)
                        second = T.if_then_else(hi < valid_limit, hi, lo)
                        stride = T.alloc_var("int64", init=T.Cast("int64", second - lo))
                        dst_pairs = T.StridedTensor(
                            (nk, 2, 512),
                            (512, stride * 512, 1),
                            "float32",
                            data=dKV_accum.data,
                        )
                        for _ in T.Serial(1):
                            T.assume(n_burst > 0)
                            T.assume(n_burst == 0 or safe_lo < dst_pairs.shape[0])
                            T.assume_no_conflict(dKV_accum)
                            T.copy(dkv_full[i * 2 : i * 2 + n_burst, :], dst_pairs[safe_lo, :n_burst, :])
                    T.set_atomic_none()

                # 9. After all nb KV tiles, drain cdq -> pack BF16 into dq_final -> store dQ[qidx].
                for d in T.Pipelined(4, num_stages=2):
                    T.dual_copy(cdq[d, :, :], chunk_fp32)
                    with T.SimdVF():
                        mask16 = T.simd.pset(16)
                        mask32 = T.simd.pset(32)
                        for r in range(16):
                            packed = T.alloc_local((2, 2), "bfloat16x128")
                            for rr in T.unroll(2):
                                for cc in T.unroll(2):
                                    x = T.simd.vld(chunk_fp32[r * 2 + rr, cc * 64])
                                    if rr == 0:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=0)
                                    else:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=1)
                            m0 = T.simd.vadd(packed[0, 0], packed[1, 0], mask16)
                            m1 = T.simd.vadd(packed[0, 1], packed[1, 1], mask16)
                            row0, row1 = T.simd.vdintlv(m0, m1)
                            T.simd.vsts(dq_final[r * 2, d * 128], row0, mask16)
                            T.simd.vsts(dq_final[r * 2 + 1, d * 128], row1, mask16)
                    T.copy(
                        dq_final[:, d * 128 : (d + 1) * 128],
                        dQ[qidx, sid * 32 : sid * 32 + 32, d * 128 : (d + 1) * 128],
                    )

    return main
