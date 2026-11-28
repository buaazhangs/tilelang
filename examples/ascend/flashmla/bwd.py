"""Compilation entry points and standalone verification/benchmark runner for FlashMLA backward."""

import argparse
from pathlib import Path

import torch
import tilelang

from examples.ascend.flashmla.bwd_kernel import (
    sparse_attention_bwd_main,
    sparse_attention_bwd_postprocess,
    sparse_attention_bwd_preprocess,
)
from examples.ascend.flashmla.bwd_kernel_v2 import (
    sparse_attention_bwd_main_v2,
    sparse_attention_bwd_postprocess_v2,
    sparse_attention_bwd_preprocess_v2,
)
from examples.ascend.flashmla.bwd_kernel_v3 import (
    sparse_attention_bwd_main_v3,
    sparse_attention_bwd_postprocess_v3,
    sparse_attention_bwd_preprocess_v3,
)
from examples.ascend.flashmla.bwd_kernel_v8 import (
    sparse_attention_bwd_main_v8,
    sparse_attention_bwd_postprocess_v8,
    sparse_attention_bwd_preprocess_v8,
)
from examples.ascend.flashmla.bwd_kernel_v14 import (
    sparse_attention_bwd_main_v14,
    sparse_attention_bwd_postprocess_v14,
    sparse_attention_bwd_preprocess_v14,
)
from examples.ascend.flashmla.bwd_kernel_v17 import (
    sparse_attention_bwd_main_v17,
    sparse_attention_bwd_postprocess_v17,
    sparse_attention_bwd_preprocess_v17,
)

_BUILDERS = {
    "v1": (sparse_attention_bwd_preprocess, sparse_attention_bwd_main, sparse_attention_bwd_postprocess),
    "v2": (sparse_attention_bwd_preprocess_v2, sparse_attention_bwd_main_v2, sparse_attention_bwd_postprocess_v2),
    "v3": (sparse_attention_bwd_preprocess_v3, sparse_attention_bwd_main_v3, sparse_attention_bwd_postprocess_v3),
    "v8": (sparse_attention_bwd_preprocess_v8, sparse_attention_bwd_main_v8, sparse_attention_bwd_postprocess_v8),
    "v14": (sparse_attention_bwd_preprocess_v14, sparse_attention_bwd_main_v14, sparse_attention_bwd_postprocess_v14),
    "v17": (sparse_attention_bwd_preprocess_v17, sparse_attention_bwd_main_v17, sparse_attention_bwd_postprocess_v17),
}


def _resolve_core_count(core_count):
    if core_count is not None:
        return core_count
    import torch_npu  # noqa: F401

    return torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num


def compile_bwd_preprocess(nq, *, sink=True, core_count=None, version="v3"):
    core_count = _resolve_core_count(core_count)
    builder = _BUILDERS[version][0]
    return tilelang.compile(
        builder(nq, sink=sink, core_count=core_count),
        target="ascend",
        out_idx=None,
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )


def compile_bwd_main(
    nq,
    nk,
    topk,
    *,
    scale=512 ** (-0.5),
    q_strides=None,
    index_stride=None,
    core_count=None,
    version="v3",
):
    core_count = _resolve_core_count(core_count)
    builder = _BUILDERS[version][1]
    return tilelang.compile(
        builder(
            nq,
            nk,
            topk,
            scale=scale,
            q_strides=q_strides,
            index_stride=index_stride,
            core_count=core_count,
        ),
        target="ascend",
        out_idx=None,
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )


def compile_bwd_postprocess(nk, *, core_count=None, version="v3"):
    core_count = _resolve_core_count(core_count)
    builder = _BUILDERS[version][2]
    return tilelang.compile(
        builder(nk, core_count=core_count),
        target="ascend",
        out_idx=None,
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )


def ref_sparse_mla_fwd_and_bwd_cpu(q_bf16, kv_bf16, indices_i32, do_bf16, sink_fp32, scale, *, sink=True):
    """Exact FP32 CPU autograd reference matching FlashMLA forward (kernel.py) + backward."""
    nq, hq, d = q_bf16.shape
    nk, _ = kv_bf16.shape
    assert hq == 64 and d == 512

    q = q_bf16.float().detach().clone().requires_grad_(True)
    kv = kv_bf16.float().detach().clone().requires_grad_(True)
    s_param = sink_fp32.float().detach().clone().requires_grad_(True) if sink else None
    do = do_bf16.float().detach().clone()

    invalid = (indices_i32 < 0) | (indices_i32 >= nk)  # [nq, topk]
    safe_idx = indices_i32.masked_fill(invalid, 0).long()  # [nq, topk]
    valid_any = (~invalid).any(dim=-1)  # [nq]

    focused_kv = kv[safe_idx]  # [nq, topk, 512]
    logits = torch.matmul(q, focused_kv.transpose(-1, -2)) * scale  # [nq, 64, topk]
    logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))

    # Match kernel.py: lse is logsumexp WITHOUT sink; all-invalid rows have lse=+inf, out=0.
    lse_raw = torch.logsumexp(logits, dim=-1)  # [nq, 64]
    lse_out = torch.where(valid_any[:, None], lse_raw, torch.full_like(lse_raw, float("inf")))

    if sink:
        sink_col = s_param.view(1, 64, 1).expand(nq, 64, 1)
        logits_ext = torch.cat([logits, sink_col], dim=-1)  # [nq, 64, topk + 1]
        probs_ext = torch.softmax(logits_ext, dim=-1)
        probs = probs_ext[..., :-1]
        # Zero out all-invalid rows to match kernel.py (denorm=0 when invalid).
        probs = torch.where(valid_any[:, None, None], probs, torch.zeros_like(probs))
    else:
        probs = torch.softmax(logits, dim=-1)
        probs = torch.where(valid_any[:, None, None], probs, torch.zeros_like(probs))

    out = torch.matmul(probs, focused_kv)  # [nq, 64, 512]
    out.backward(do)

    ref_dq = q.grad.detach()
    ref_dkv = kv.grad.detach()
    ref_dsink = s_param.grad.detach() if sink else torch.zeros(64, dtype=torch.float32)
    return out.detach().to(torch.bfloat16), lse_out.detach(), ref_dq, ref_dkv, ref_dsink


def make_test_inputs_cpu(nq, nk, topk, *, seed=42, causal_like=True, include_all_invalid=False):
    """Generate test inputs purely on CPU (safe & fast for both npusim and onboard)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((nq, 64, 512), generator=g, dtype=torch.float32).to(torch.bfloat16)
    kv = torch.randn((nk, 512), generator=g, dtype=torch.float32).to(torch.bfloat16)
    do = torch.randn((nq, 64, 512), generator=g, dtype=torch.float32).to(torch.bfloat16)
    sink = (torch.randn((64,), generator=g, dtype=torch.float32) * 0.5).contiguous()

    indices = torch.full((nq, topk), -1, dtype=torch.int32)
    for i in range(nq):
        if include_all_invalid and i == nq - 1:
            continue  # leave all -1 to test all-invalid row handling
        if causal_like:
            # Allow causal-like upper bound with plenty of valid tokens and some -1 tails.
            upper = min(nk, max(1, (i + 1) * max(1, nk // max(nq, 1))))
            n_valid = min(topk, upper)
            perm = torch.randperm(upper, generator=g)[:n_valid].to(torch.int32)
            indices[i, :n_valid] = perm
        else:
            perm = torch.randperm(nk, generator=g)[: min(topk, nk)].to(torch.int32)
            indices[i, : perm.numel()] = perm
            if topk > 64:
                indices[i, ::37] = -1
    return q.contiguous(), kv.contiguous(), indices.contiguous(), do.contiguous(), sink.contiguous()


def check_close(name, actual, expected, *, cos_tol=1e-3, max_abs_tol=5e-2, rel_tol=2e-2):
    a = actual.cpu().float()
    e = expected.cpu().float()
    diff = (a - e).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = e.abs().max().item() + 1e-8
    rel_max = max_abs / denom
    a_flat = a.reshape(-1).double()
    e_flat = e.reshape(-1).double()
    na = torch.dot(a_flat, a_flat).sqrt().item()
    ne = torch.dot(e_flat, e_flat).sqrt().item()
    if na == 0.0 and ne == 0.0:
        cos_diff = 0.0
    else:
        cos_diff = 1.0 - (torch.dot(a_flat, e_flat).item() / (na * ne + 1e-30))
    print(
        f"[{name}] max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rel_max={rel_max:.6f} cos_diff={cos_diff:.6e}",
        flush=True,
    )
    assert cos_diff < cos_tol or (max_abs < max_abs_tol and rel_max < rel_tol), (
        f"{name} mismatch: max_abs={max_abs}, rel_max={rel_max}, cos_diff={cos_diff}"
    )


def ref_sparse_mla_fwd_and_bwd_chunked(
    q_bf16, kv_bf16, indices_i32, do_bf16, sink_fp32, scale, *, sink=True, chunk_q=64
):
    """Chunked FP32 exact reference (fast on NPU even for nq=4096, nk=4096, topk=640)."""
    dev = q_bf16.device
    nq = q_bf16.shape[0]
    nk = kv_bf16.shape[0]
    kv_f = kv_bf16.float()
    s_f = sink_fp32.float()

    out_bf16 = torch.empty_like(q_bf16)
    lse_out = torch.empty((nq, 64), dtype=torch.float32, device=dev)
    ref_dq = torch.empty((nq, 64, 512), dtype=torch.float32, device=dev)
    ref_dkv = torch.zeros((nk, 512), dtype=torch.float32, device=dev)
    ref_dsink = torch.zeros((64,), dtype=torch.float32, device=dev)

    for start in range(0, nq, chunk_q):
        end = min(nq, start + chunk_q)
        qc = q_bf16[start:end].float()
        doc = do_bf16[start:end].float()
        idxc = indices_i32[start:end]

        invalid = (idxc < 0) | (idxc >= nk)
        safe_idx = idxc.masked_fill(invalid, 0).long()
        valid_any = (~invalid).any(dim=-1)

        fkv = kv_f[safe_idx]  # [c, topk, 512]
        logits = torch.matmul(qc, fkv.transpose(-1, -2)) * scale  # [c, 64, topk]
        logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))

        lse_raw = torch.logsumexp(logits, dim=-1)
        lse_c = torch.where(valid_any[:, None], lse_raw, torch.full_like(lse_raw, float("inf")))
        lse_out[start:end] = lse_c

        if sink:
            lse_tot = torch.logaddexp(lse_raw, s_f.view(1, 64))
            lse_tot = torch.where(valid_any[:, None], lse_tot, torch.full_like(lse_tot, float("inf")))
            probs = torch.exp(logits - lse_tot.unsqueeze(-1))
            probs = torch.where(valid_any[:, None, None], probs, torch.zeros_like(probs))
            p_sink = torch.where(valid_any[:, None], torch.exp(s_f.view(1, 64) - lse_tot), torch.zeros_like(lse_tot))
        else:
            probs = torch.softmax(logits, dim=-1)
            probs = torch.where(valid_any[:, None, None], probs, torch.zeros_like(probs))

        oc = torch.matmul(probs, fkv)
        out_bf16[start:end] = oc.to(torch.bfloat16)

        # Exact backward using BF16 O (matching what preprocess receives)
        delta_c = (oc.to(torch.bfloat16).float() * doc).sum(dim=-1)  # [c, 64]
        dp = torch.matmul(doc, fkv.transpose(-1, -2))  # [c, 64, topk]
        ds = probs * (dp - delta_c.unsqueeze(-1)) * scale
        ref_dq[start:end] = torch.matmul(ds, fkv)
        dkv_local = torch.matmul(probs.transpose(-1, -2), doc) + torch.matmul(ds.transpose(-1, -2), qc)  # [c, topk, 512]
        dkv_local = dkv_local.masked_fill(invalid.unsqueeze(-1), 0.0)
        ref_dkv.index_add_(0, safe_idx.reshape(-1), dkv_local.reshape(-1, 512))
        if sink:
            ref_dsink -= (p_sink * delta_c).sum(dim=0)

    return out_bf16, lse_out, ref_dq, ref_dkv, ref_dsink


def _profile_kernel_onboard(name, fn, prof_dir: Path, flush_buf, *, warmup=3, repeat=5):
    import csv
    import shutil
    import torch_npu.profiler as prof

    out_dir = prof_dir / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(warmup):
        flush_buf.zero_()
        fn()
    torch.npu.synchronize()

    with prof.profile(
        activities=[prof.ProfilerActivity.NPU],
        schedule=prof.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=prof.tensorboard_trace_handler(str(out_dir)),
        experimental_config=prof._ExperimentalConfig(
            profiler_level=prof.ProfilerLevel.Level1,
            aic_metrics=prof.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as trace:
        for _ in range(repeat):
            flush_buf.zero_()
            fn()
        torch.npu.synchronize()
        trace.step()

    csvs = list(out_dir.rglob("kernel_details.csv"))
    assert csvs, f"No kernel_details.csv found under {out_dir}"
    rows = []
    with csvs[0].open() as f:
        for r in csv.DictReader(f):
            if r.get("Name") == "main_kernel":
                rows.append(r)
    assert rows, f"No main_kernel rows in {csvs[0]}"
    keys = [
        "Duration(us)",
        "aic_mac_ratio",
        "aic_scalar_ratio",
        "aic_mte1_ratio",
        "aic_mte2_ratio",
        "aic_fixpipe_ratio",
        "aiv_vec_ratio",
        "aiv_scalar_ratio",
        "aiv_mte2_ratio",
        "aiv_mte3_ratio",
        "cube_utilization(%)",
    ]
    summary = {k: sum(float(r[k]) for r in rows if r.get(k) not in (None, "", "N/A")) / len(rows) for k in keys}
    summary["samples"] = len(rows)
    summary["csv"] = str(csvs[0])
    traces = list(out_dir.rglob("trace_view.json"))
    summary["trace"] = str(traces[0]) if traces else ""
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compile", "verify", "bench"], default="verify")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v8", "v14", "v17"], default="v14")
    parser.add_argument("--nq", type=int, default=4)
    parser.add_argument("--nk", type=int, default=128)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--core-count", type=int, default=36)
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--all-invalid", action="store_true")
    parser.add_argument("--ref-device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--prof-dir", type=Path, default=Path("/home/sym/zsm/runs/flashmla-bwd-prof"))
    args = parser.parse_args()

    sink_enabled = not args.no_sink
    scale = 512 ** (-0.5)

    k_pre = compile_bwd_preprocess(args.nq, sink=sink_enabled, core_count=args.core_count, version=args.version)
    k_main = compile_bwd_main(
        args.nq, args.nk, args.topk, scale=scale, core_count=args.core_count, version=args.version
    )
    k_post = compile_bwd_postprocess(args.nk, core_count=args.core_count, version=args.version)

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        (args.dump_dir / f"{args.version}_bwd_preprocess.asc").write_text(k_pre.get_kernel_source())
        (args.dump_dir / f"{args.version}_bwd_main.asc").write_text(k_main.get_kernel_source())
        (args.dump_dir / f"{args.version}_bwd_postprocess.asc").write_text(k_post.get_kernel_source())
        print(f"Dumped generated AscendC sources to {args.dump_dir}", flush=True)

    if args.mode == "compile":
        print(f"Compile-only ({args.version}) succeeded.", flush=True)
        return

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(120000)

    if args.mode == "verify":
        q_cpu, kv_cpu, idx_cpu, do_cpu, sink_cpu = make_test_inputs_cpu(
            args.nq,
            args.nk,
            args.topk,
            causal_like=True,
            include_all_invalid=args.all_invalid,
        )
        if args.ref_device == "npu":
            q = q_cpu.to("npu")
            kv = kv_cpu.to("npu")
            idx = idx_cpu.to("npu")
            do = do_cpu.to("npu")
            sink_t = sink_cpu.to("npu")
            o, lse, ref_dq, ref_dkv, ref_dsink = ref_sparse_mla_fwd_and_bwd_chunked(
                q, kv, idx, do, sink_t, scale, sink=sink_enabled
            )
            torch.npu.synchronize()
        else:
            o_cpu, lse_cpu, ref_dq, ref_dkv, ref_dsink = ref_sparse_mla_fwd_and_bwd_cpu(
                q_cpu, kv_cpu, idx_cpu, do_cpu, sink_cpu, scale, sink=sink_enabled
            )
            q, kv, idx, o, do, lse, sink_t = (
                q_cpu.to("npu"),
                kv_cpu.to("npu"),
                idx_cpu.to("npu"),
                o_cpu.to("npu"),
                do_cpu.to("npu"),
                lse_cpu.to("npu"),
                sink_cpu.to("npu"),
            )

        delta = torch.zeros((args.nq, 64), dtype=torch.float32, device="npu")
        lse_tot = torch.zeros((args.nq, 64), dtype=torch.float32, device="npu")
        dsink = torch.zeros((64,), dtype=torch.float32, device="npu")
        dq = torch.zeros((args.nq, 64, 512), dtype=torch.bfloat16, device="npu")
        dkv_accum = torch.zeros((args.nk, 512), dtype=torch.float32, device="npu")
        dkv = torch.zeros((args.nk, 512), dtype=torch.bfloat16, device="npu")

        k_pre(o, do, lse, sink_t, delta, lse_tot, dsink)
        k_main(q, kv, do, idx, lse_tot, delta, dq, dkv_accum)
        k_post(dkv_accum, dkv)
        torch.npu.synchronize()

        check_close("dQ", dq, ref_dq, cos_tol=1e-3, max_abs_tol=5e-2, rel_tol=3e-2)
        check_close("dKV", dkv, ref_dkv, cos_tol=1e-3, max_abs_tol=5e-2, rel_tol=3e-2)
        if sink_enabled:
            check_close("dSink", dsink, ref_dsink, cos_tol=1e-4, max_abs_tol=5e-2, rel_tol=5e-3)
        print(f"BWD_VERIFY_PASS ({args.version}, nq={args.nq}, nk={args.nk}, topk={args.topk})", flush=True)
        return

    # Onboard benchmark mode using torch_npu.profiler (Level1 + PipeUtilization).
    import json

    torch.manual_seed(42)
    q = torch.randn((args.nq, 64, 512), dtype=torch.bfloat16, device="npu")
    kv = torch.randn((args.nk, 512), dtype=torch.bfloat16, device="npu")
    o = torch.randn((args.nq, 64, 512), dtype=torch.bfloat16, device="npu")
    do = torch.randn((args.nq, 64, 512), dtype=torch.bfloat16, device="npu")
    lse = torch.randn((args.nq, 64), dtype=torch.float32, device="npu") + 5.0
    sink_t = torch.zeros((64,), dtype=torch.float32, device="npu")
    # Realistic sparse indices across full nk range
    idx = torch.randint(0, args.nk, (args.nq, args.topk), dtype=torch.int32, device="npu")

    delta = torch.empty((args.nq, 64), dtype=torch.float32, device="npu")
    lse_tot = torch.empty((args.nq, 64), dtype=torch.float32, device="npu")
    dsink = torch.zeros((64,), dtype=torch.float32, device="npu")
    dq = torch.empty((args.nq, 64, 512), dtype=torch.bfloat16, device="npu")
    dkv_accum = torch.zeros((args.nk, 512), dtype=torch.float32, device="npu")
    dkv = torch.empty((args.nk, 512), dtype=torch.bfloat16, device="npu")
    flush_buf = torch.empty(256_000_000, dtype=torch.int32, device="npu")

    k_pre(o, do, lse, sink_t, delta, lse_tot, dsink)
    torch.npu.synchronize()

    tag = f"{args.version}_{args.nq}_{args.nk}_{args.topk}_c{args.core_count}"
    s_pre = _profile_kernel_onboard(f"{tag}_pre", lambda: (dsink.zero_(), k_pre(o, do, lse, sink_t, delta, lse_tot, dsink)), args.prof_dir, flush_buf)
    s_main = _profile_kernel_onboard(f"{tag}_main", lambda: (dkv_accum.zero_(), k_main(q, kv, do, idx, lse_tot, delta, dq, dkv_accum)), args.prof_dir, flush_buf)
    s_post = _profile_kernel_onboard(f"{tag}_post", lambda: k_post(dkv_accum, dkv), args.prof_dir, flush_buf)

    flops = 10.0 * args.nq * 64 * args.topk * 512
    peak_tflops = args.core_count * 13.5168
    main_us = s_main["Duration(us)"]
    e2e_us = s_pre["Duration(us)"] + main_us + s_post["Duration(us)"]
    main_tflops = flops / (main_us * 1e6)
    e2e_tflops = flops / (e2e_us * 1e6)
    report = {
        "version": args.version,
        "shape": {"nq": args.nq, "nk": args.nk, "topk": args.topk, "core_count": args.core_count},
        "flops_g": flops / 1e9,
        "peak_tflops": peak_tflops,
        "main_us": main_us,
        "main_tflops": main_tflops,
        "main_mfu_pct": main_tflops / peak_tflops * 100.0,
        "e2e_us": e2e_us,
        "e2e_tflops": e2e_tflops,
        "e2e_mfu_pct": e2e_tflops / peak_tflops * 100.0,
        "pre": s_pre,
        "main": s_main,
        "post": s_post,
    }
    print(f"=== BENCH SUMMARY ({tag}) ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
