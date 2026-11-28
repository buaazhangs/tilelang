"""v8: BLOCK_N=128 with FULL double buffering (aq:2, bq:2, bv:2, cdkv:2, kv_l1:2, pds_l1:2).

Fixes v7's single-buffered MTE1<->M<->FIX serialization (which inflated aic_fixpipe to 86.6%)
by using 64-wide inner chunks (8 steps over D=512) so L0A, L0B, and L0C all fit 2 versions
within exact hardware limits while keeping 128-wide KV tiles (2x larger M/N/K than v3/v4a):
  - L1:  q_l1(64K) + do_l1(64K) + pds_l1:2(64K) + kv_l1_0:2(128K) + kv_l1_1:2(128K) = 448 KiB
  - L0A: aq:2(64x64, 16K) + ads:1(64x128, 16K) + ap:1(128x64, 16K) + ads_t:1(128x64, 16K) = 64 KiB
  - L0B: bq:2(128x64, 32K, Stage 0) + bv:2(128x64, 32K, Stage 1) = 64 KiB
  - L0C: cdq(8x64x64, 128K) + cs(64x128, 32K) + cdp(64x128, 32K) + cdkv:2(128x64, 64K) = 256 KiB
  - UB:  ~244 KiB <= 256 KiB
"""

import tilelang.ascend.language as T
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
)
from examples.ascend.flashmla.bwd_kernel_v2 import (
    sparse_attention_bwd_postprocess_v2,
    sparse_attention_bwd_preprocess_v2,
)
from examples.ascend.flashmla.bwd_kernel_v14 import sparse_attention_bwd_main_v14

sparse_attention_bwd_preprocess_v8 = sparse_attention_bwd_preprocess_v2
sparse_attention_bwd_postprocess_v8 = sparse_attention_bwd_postprocess_v2


def sparse_attention_bwd_main_v8(
    nq,
    nk,
    topk,
    *,
    scale=512 ** (-0.5),
    q_strides=None,
    index_stride=None,
    core_count=36,
):
    """v8 full-double-buffered BLOCK_N=128 FlashMLA backward main mixed kernel."""
    if nq <= 0 or nk <= 0 or topk <= 0 or topk % 64:
        raise ValueError("nq/nk must be positive and topk a positive multiple of 64")
    if core_count <= 0:
        raise ValueError("core_count must be a positive integer")
    if topk % 128 != 0:
        return sparse_attention_bwd_main_v14(
            nq, nk, topk, scale=scale, q_strides=q_strides, index_stride=index_stride, core_count=core_count
        )

    q_strides = q_strides or (64 * 512, 512, 1)
    index_stride = index_stride or topk
    nb = topk // 128
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
            # L1: Bank 0 (192 KiB) + Bank 1 (256 KiB) = 448 KiB <= 512 KiB
            q_l1 = T.alloc_l1((64, 512), "bfloat16")
            do_l1 = T.alloc_l1((64, 512), "bfloat16")
            pds_l1 = T.alloc_l1((2, 64, 128), "bfloat16")
            kv_l1_0 = T.alloc_l1((64, 512), "bfloat16")
            kv_l1_1 = T.alloc_l1((64, 512), "bfloat16")

            # L0A: aq:2 (16K) + ads:1 (16K) + ap:1 (16K) + ads_t:1 (16K) = 64 KiB
            aq = T.alloc_l0a((64, 64), "bfloat16")
            ads = T.alloc_l0a((64, 128), "bfloat16")
            ap = T.alloc_l0a((128, 64), "bfloat16")
            ads_t = T.alloc_l0a((128, 64), "bfloat16")

            # L0B: bq:2 (32K, Stage 0) + bv:2 (32K, Stage 1) = 64 KiB
            bq = T.alloc_l0b((128, 64), "bfloat16")
            bv = T.alloc_l0b((128, 64), "bfloat16")
            bv_sub = T.view(bv, shape=(2, 64, 64), dtype="bfloat16")

            # L0C: cdq (128K) + cs (32K) + cdp (32K) + cdkv:2 (64K) = 256 KiB
            cdq = T.alloc_l0c((8, 64, 64), "float32")
            cs = T.alloc_l0c((64, 128), "float32")
            cdp = T.alloc_l0c((64, 128), "float32")
            cdkv = T.alloc_l0c((128, 64), "float32")

            # UB per AIV (~244 KiB <= 256 KiB)
            kv_ub = T.alloc_shared((32, 512), "bfloat16")
            nz_ub = T.alloc_shared((33, 512), "bfloat16")
            s_ub = T.alloc_shared((32, 128), "float32")
            dp_ub = T.alloc_shared((32, 128), "float32")
            p_ub = T.alloc_shared((32, 128), "bfloat16")
            ds_ub = T.alloc_shared((32, 128), "bfloat16")
            dkv_full = T.alloc_shared((64, 512), "float32")
            dq_final = T.view(dkv_full, shape=(64, 1024), dtype="bfloat16")
            lse_ub = T.alloc_shared((32,), "float32")
            delta_ub = T.alloc_shared((32,), "float32")
            lse_rep = T.alloc_shared((64,), "float32")
            delta_rep = T.alloc_shared((64,), "float32")
            idx_ub = T.alloc_shared((128,), "int32")
            smaller = T.alloc_shared((64,), "uint32")
            larger = T.alloc_shared((64,), "uint32")

            T.annotate_buffer_addresses(
                {
                    q_l1: 0,
                    do_l1: 65536,
                    pds_l1: 131072,
                    kv_l1_0: 262144,
                    kv_l1_1: 393216,
                }
            )
            T.annotate_layout(
                {
                    q_l1: make_ascend_major_k_layout(q_l1),
                    do_l1: make_ascend_major_k_layout(do_l1),
                    kv_l1_0: make_ascend_major_k_layout(kv_l1_0),
                    kv_l1_1: make_ascend_major_k_layout(kv_l1_1),
                    pds_l1: make_ascend_major_k_layout(pds_l1),
                    aq: make_ascend_major_k_layout(aq),
                    ads: make_ascend_major_k_layout(ads),
                    ap: make_ascend_major_k_layout(ap),
                    ads_t: make_ascend_major_k_layout(ads_t),
                    bq: make_ascend_major_k_layout(bq),
                    bv: make_ascend_major_mn_layout(bv),
                    bv_sub: make_ascend_major_mn_layout(bv_sub),
                    nz_ub: make_ascend_compact_nz_layout(nz_ub),
                    s_ub: make_ascend_l0c_layout(s_ub),
                    dp_ub: make_ascend_l0c_layout(dp_ub),
                    p_ub: make_ascend_compact_nz_layout(p_ub),
                    ds_ub: make_ascend_compact_nz_layout(ds_ub),
                }
            )
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    do_l1: 1,
                    kv_l1_0: 2,
                    kv_l1_1: 2,
                    pds_l1: 2,
                    aq: 2,
                    ads: 1,
                    ap: 1,
                    ads_t: 1,
                    bq: 2,
                    bv: 2,
                    cs: 1,
                    cdp: 1,
                    cdq: 1,
                    cdkv: 2,
                    kv_ub: 1,
                    nz_ub: 1,
                    s_ub: 1,
                    dp_ub: 1,
                    p_ub: 1,
                    ds_ub: 1,
                    dkv_full: 1,
                    lse_ub: 1,
                    delta_ub: 1,
                    lse_rep: 1,
                    delta_rep: 1,
                    idx_ub: 1,
                    smaller: 2,
                    larger: 2,
                }
            )
            with T.SimdVF():
                T.fill(kv_ub, 0)

            for job in T.Serial(T.ceildiv(nq - bx, num_blocks)):
                qidx = bx + job * num_blocks

                T.copy(Q[qidx, :, :], q_l1)
                T.copy(dO[qidx, :, :], do_l1)
                T.copy(LSE_tot[qidx, sid * 32 : sid * 32 + 32], lse_ub)
                T.copy(Delta[qidx, sid * 32 : sid * 32 + 32], delta_ub)
                with T.SimdVF():
                    mask32 = T.simd.pset(32)
                    l_v = T.simd.vld(lse_ub[0])
                    d_v = T.simd.vld(delta_ub[0])
                    l_rep, _ig0 = T.simd.vintlv(l_v, l_v)
                    d_rep, _ig1 = T.simd.vintlv(d_v, d_v)
                    T.simd.vsts(lse_rep[0], l_rep, mask32)
                    T.simd.vsts(delta_rep[0], d_rep, mask32)

                for k in T.Pipelined(nb, num_stages=2, annotations={"enable_offset": True}):
                    # Stage 0: gather2 + ND->NZ -> kv_l1_0/1 + S + dP + P/dS -> pds_l1.
                    with T.Stage(0):
                        T.copy(Indices[qidx, k * 128 : (k + 1) * 128], idx_ub)
                        with T.SimdVF():
                            pairs32 = T.simd.pset(32, "PAT_VL32")
                            for h in T.unroll(2):
                                x = T.reinterpret("uint32x64", T.simd.vld(idx_ub[h * 64]))
                                even, odd = T.simd.vdintlv(x, x)
                                lo = T.simd.vmins(T.simd.vmin(even, odd), valid_limit)
                                hi = T.simd.vmax(even, odd)
                                T.simd.vsts(smaller[h * 32], lo, pairs32, extent=32)
                                T.simd.vsts(larger[h * 32], hi, pairs32, extent=32)

                        for sub in T.Serial(2):
                            for i in T.unroll(16):
                                lo = smaller[sid * 32 + sub * 16 + i]
                                hi = larger[sid * 32 + sub * 16 + i]
                                second = T.if_then_else(hi < valid_limit, hi, lo)
                                n_burst = T.alloc_var("int32", init=T.if_then_else(lo < valid_limit, 2, 0))
                                safe_lo = T.if_then_else(n_burst == 0, T.uint32(0), lo)
                                stride = T.alloc_var("int64", init=T.Cast("int64", second - lo))
                                main_pairs = T.StridedTensor(
                                    (nk, 2, 512), (512, stride * 512, 1), "bfloat16", data=KV.data
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
                                        x1, src_ptr = T.simd.vld(
                                            src_ptr, post_inc=T.if_then_else(row == 31, -15744, 384)
                                        )
                                        config = (33 << 16) | T.if_then_else(row == 31, 497, 1)
                                        dst0 = T.simd.vsstb(x0, dst0, config, nd_mask, update=True)
                                        dst1 = T.simd.vsstb(x1, dst1, config, nd_mask, update=True)

                            if sid == 0:
                                T.copy(nz_ub[:32, :], kv_l1_0[sub * 32 : (sub + 1) * 32, :])
                            else:
                                T.copy(nz_ub[:32, :], kv_l1_1[sub * 32 : (sub + 1) * 32, :])

                        T.set_mmad_direction("m")
                        for d in T.Pipelined(8, num_stages=2):
                            T.copy(q_l1[:, d * 64 : (d + 1) * 64], aq)
                            with T.Task():
                                T.copy(kv_l1_0[:, d * 64 : (d + 1) * 64], bq[:64, :], l0_pitch=8)
                                T.copy(kv_l1_1[:, d * 64 : (d + 1) * 64], bq[64:, :], l0_pitch=8)
                            T.gemm(
                                aq, bq, cs, transpose_B=True, clear_accum=d == 0,
                                unit_flag_ctrl=T.if_then_else(d == 7, 3, 2),
                            )
                        T.dual_copy(cs, s_ub, unit_flag_ctrl=3)

                        for d in T.Pipelined(8, num_stages=2):
                            T.copy(do_l1[:, d * 64 : (d + 1) * 64], aq)
                            with T.Task():
                                T.copy(kv_l1_0[:, d * 64 : (d + 1) * 64], bq[:64, :], l0_pitch=8)
                                T.copy(kv_l1_1[:, d * 64 : (d + 1) * 64], bq[64:, :], l0_pitch=8)
                            T.gemm(
                                aq, bq, cdp, transpose_B=True, clear_accum=d == 0,
                                unit_flag_ctrl=T.if_then_else(d == 7, 3, 2),
                            )
                        T.dual_copy(cdp, dp_ub, unit_flag_ctrl=3)

                        with T.SimdVF():
                            full = T.simd.pset(32)
                            masks = T.alloc_local((8,), "boolx256")
                            for col in T.unroll(8):
                                lo = T.simd.vld(T.access_ptr(smaller[col * 8], "r", extent=8), "BLK")
                                hi = T.simd.vld(T.access_ptr(larger[col * 8], "r", extent=8), "BLK")
                                paired, _ig = T.simd.vintlv(lo, hi)
                                masks[col] = T.simd.vcmps(paired, valid_limit, op="lt")

                            for r in T.unroll(4, unroll_factor=1):
                                l_row = T.alloc_local((2,), "float32x64")
                                d_row = T.alloc_local((2,), "float32x64")
                                row_valid = T.alloc_local((2,), "boolx256")
                                p_packed = T.alloc_local((2, 2), "bfloat16x128")
                                ds_packed = T.alloc_local((2, 2), "bfloat16x128")
                                for rr in T.unroll(2):
                                    l_row[rr] = T.simd.vld(lse_rep[r * 16 + rr * 8], "E2B_B32")
                                    d_row[rr] = T.simd.vld(delta_rep[r * 16 + rr * 8], "E2B_B32")
                                    row_valid[rr] = T.simd.vcmps(l_row[rr], T.float32(1e30), full, op="lt")
                                for pair in T.unroll(4):
                                    for cc in T.unroll(2):
                                        for rr in T.unroll(2):
                                            vmask = T.simd.pand(masks[pair * 2 + cc], row_valid[rr], full)
                                            s_x = T.simd.vld(s_ub[r * 8 + rr * 4, (pair * 2 + cc) * 16])
                                            dp_x = T.simd.vld(dp_ub[r * 8 + rr * 4, (pair * 2 + cc) * 16])
                                            prob = T.simd.vexpdif(
                                                T.simd.vmuls(s_x, T.float32(scale), full), l_row[rr], vmask
                                            )
                                            ds_val = T.simd.vmuls(
                                                T.simd.vmul(prob, T.simd.vsub(dp_x, d_row[rr], full), vmask),
                                                T.float32(scale), vmask,
                                            )
                                            if cc == 0:
                                                p_packed[rr, cc] = T.simd.vcvt(prob, "bfloat16", full, part=0)
                                                ds_packed[rr, cc] = T.simd.vcvt(ds_val, "bfloat16", full, part=0)
                                            else:
                                                p_packed[rr, cc] = T.simd.vcvt(prob, "bfloat16", full, part=1)
                                                ds_packed[rr, cc] = T.simd.vcvt(ds_val, "bfloat16", full, part=1)
                                    pm0 = T.simd.vadd(p_packed[0, 0], p_packed[0, 1])
                                    pm1 = T.simd.vadd(p_packed[1, 0], p_packed[1, 1])
                                    pcol0, pcol1 = T.simd.vdintlv(pm0, pm1)
                                    T.simd.vsts(p_ub[r * 8, pair * 32], pcol0)
                                    T.simd.vsts(p_ub[r * 8, pair * 32 + 16], pcol1)

                                    dm0 = T.simd.vadd(ds_packed[0, 0], ds_packed[0, 1])
                                    dm1 = T.simd.vadd(ds_packed[1, 0], ds_packed[1, 1])
                                    dcol0, dcol1 = T.simd.vdintlv(dm0, dm1)
                                    T.simd.vsts(ds_ub[r * 8, pair * 32], dcol0)
                                    T.simd.vsts(ds_ub[r * 8, pair * 32 + 16], dcol1)

                        T.copy(p_ub, pds_l1[0, sid * 32 : sid * 32 + 32, :])
                        T.copy(ds_ub, pds_l1[1, sid * 32 : sid * 32 + 32, :])

                    # Stage 1: dQ (64x128x64, bv:2) + dV+dK (128x64x64, bv:2, cdkv:2) + scatter2.
                    with T.Stage(1):
                        T.set_mmad_direction("n")
                        T.copy(pds_l1[1, :, :], ads)
                        for d in T.Pipelined(8, num_stages=2):
                            with T.Task():
                                T.copy(kv_l1_0[:, d * 64 : (d + 1) * 64], bv[:64, :], l0_pitch=4)
                                T.copy(kv_l1_1[:, d * 64 : (d + 1) * 64], bv[64:, :], l0_pitch=4)
                            T.gemm(ads, bv, cdq[d, :, :], clear_accum=k == 0)

                        T.copy(pds_l1[0, :, :], ap, transpose=True)
                        T.copy(pds_l1[1, :, :], ads_t, transpose=True)
                        for d in T.Pipelined(8, num_stages=2):
                            T.copy(do_l1[:, d * 64 : (d + 1) * 64], bv_sub[0, :, :])
                            T.gemm(ap, bv_sub[0, :, :], cdkv, clear_accum=True, unit_flag_ctrl=2)
                            T.copy(q_l1[:, d * 64 : (d + 1) * 64], bv_sub[1, :, :])
                            T.gemm(ads_t, bv_sub[1, :, :], cdkv, clear_accum=False, unit_flag_ctrl=3)
                            T.dual_copy(cdkv, dkv_full[:, d * 64 : (d + 1) * 64], unit_flag_ctrl=3)

                        T.set_atomic("add", "float32")
                        for i in T.unroll(32):
                            lo = smaller[sid * 32 + i]
                            hi = larger[sid * 32 + i]
                            n_burst = T.alloc_var(
                                "int32",
                                init=T.if_then_else(hi < valid_limit, 2, T.if_then_else(lo < valid_limit, 1, 0)),
                            )
                            safe_lo = T.if_then_else(n_burst == 0, T.uint32(0), lo)
                            second = T.if_then_else(hi < valid_limit, hi, lo)
                            stride = T.alloc_var("int64", init=T.Cast("int64", second - lo))
                            dst_pairs = T.StridedTensor(
                                (nk, 2, 512), (512, stride * 512, 1), "float32", data=dKV_accum.data
                            )
                            for _ in T.Serial(1):
                                T.assume(n_burst > 0)
                                T.assume(n_burst == 0 or safe_lo < dst_pairs.shape[0])
                                T.assume_no_conflict(dKV_accum)
                                T.copy(dkv_full[i * 2 : i * 2 + n_burst, :], dst_pairs[safe_lo, :n_burst, :])
                        T.set_atomic_none()

                # Drain cdq (8 x 64x64) -> dkv_full[:32] -> in-place BF16 pack -> store dQ[qidx].
                for d in T.Serial(8):
                    T.dual_copy(cdq[d, :, :], dkv_full[:32, d * 64 : (d + 1) * 64])
                with T.SimdVF():
                    mask16 = T.simd.pset(16)
                    mask32 = T.simd.pset(32)
                    for r in range(16):
                        for d in T.unroll(4):
                            packed = T.alloc_local((2, 2), "bfloat16x128")
                            for rr in T.unroll(2):
                                for cc in T.unroll(2):
                                    x = T.simd.vld(dkv_full[r * 2 + rr, d * 128 + cc * 64])
                                    if rr == 0:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=0)
                                    else:
                                        packed[rr, cc] = T.simd.vcvt(x, "bfloat16", mask32, part=1)
                            m0 = T.simd.vadd(packed[0, 0], packed[1, 0], mask16)
                            m1 = T.simd.vadd(packed[0, 1], packed[1, 1], mask16)
                            row0, row1 = T.simd.vdintlv(m0, m1)
                            T.simd.vsts(dq_final[r * 2, d * 128], row0, mask16)
                            T.simd.vsts(dq_final[r * 2 + 1, d * 128], row1, mask16)
                            T.simd.mem_bar("VST_VLD")
                T.copy(dq_final[:32, :512], dQ[qidx, sid * 32 : sid * 32 + 32, :])

    return main
