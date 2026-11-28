# DeepSeek V4.1 Sparse FlashMLA Backward (Ascend 950DT) 实现与优化总结

本目录汇总了 TileLang Ascend 平台上 **DeepSeek V4.1 Sparse FlashMLA 反向算子（`bwd`）** 的完整任务规格、理论 Roofline 与硬件耦合瓶颈分析、`v1 → v3 → v4a~v17`（基于 Operator Harness 闭环迭代）演进细节、真机（36 核 `Ascend950DT_9592`）精度与性能流水实测结果、去 `scatter2` 消融诊断，以及代表性版本的 Python 源码、生成的 AscendC（`.asc`）和 Profiling 数据。

---

## 一、任务介绍与锁定规格

### 1.1 目标与约束
- **目标**：在 TileLang Ascend 上实现与前向 [kernel.py](../kernel.py)（BF16 prefill）数学严格对齐的 Sparse FlashMLA 反向算子，完成多 buffer 与软件流水线优化，并在真实 36 核 `Ascend950DT_9592` 机器上通过全量 shape 正确性校验与 `PipeUtilization` 实测。
- **核心架构决定**：保持 **三阶段（3 个独立 kernel）** 结构：
  1. **`preprocess`**（纯 AIV，72 AIV）：计算 `Delta = rowsum(O * dO)`、融合 `Sink` 后的 `LSE_tot` 以及原子累加 `dSink`。
  2. **`main`**（1C2V MixedKernel，36 AIC + 72 AIV）：按 Query（`qidx`）并行，对每个稀疏 KV 块（64 tokens/tile）完成 5 个 GEMM（$S, dP, dQ, dV, dK$），输出 `dQ`（BF16）并用硬件 `T.set_atomic("add", "float32")` 将 $dK + dV$ 高精度累加至 `dKV_accum`（FP32）。
  3. **`postprocess`**（纯 AIV，72 AIV）：利用 kernel 边界完成全网格同步后，将累加完毕的 `dKV_accum`（FP32）经 SIMD `vcvt` 转为最终 `dKV`（BF16），避免数百/数千次 BF16 原子累加导致的 7-bit 尾数精度退化。

### 1.2 算子规格（DeepSeek V4.1 FlashMLA Backward）

| 类别 | 参数 | 说明 |
|---|---|---|
| **静态属性** | `num_heads_q = 64` | Q 头数 |
| | `num_heads_kv = 1` | K 与 V 共享同一张量 `KV` |
| | `head_dim = 512` | $D_{qk} = D_v = 512$ |
| | `scaling = 512^{-0.5} ≈ 0.0442` | 可配置标量 |
| | `causal = False` | 无独立因果掩码，因果与无效槽位完全由 `Indices` 控制 |
| | `sink = True` | 支持编译开关（`sink=True/False` 均已验证） |
| | `dropout = 0` | 无 dropout |
| **动态属性** | `nq ∈ [1..64k]` | Packed 展平 Q 序列总长，**无整除约束** |
| | `nk ∈ [64..128k]` | Packed 展平 KV 序列总长，**无整除约束** |
| | `topk ∈ [64..2048]` | 稀疏 KV 槽位数，满足 `topk % 64 == 0`（典型值 `640` / `2048`） |
| **输入张量** | `Q`, `O`, `dO` | `[nq, 64, 512]`, `bfloat16` |
| | `KV` | `[nk, 512]`, `bfloat16` |
| | `Indices` | `[nq, topk]`, `int32`（`< 0` 或 `>= nk` 为无效槽位） |
| | `LSE` | `[nq, 64]`, `float32`（自然对数底 $e$，不含 `Sink`；全无效行 `LSE = +inf, O = 0`） |
| | `Sink` | `[64]`, `float32` |
| **输出张量** | `dQ` | `[nq, 64, 512]`, `bfloat16` |
| | `dKV` | `[nk, 512]`, `bfloat16`（内部 `dKV_accum` 为 `[nk, 512]`, `float32`） |
| | `dSink` | `[64]`, `float32` |

---

## 二、硬件平台与理论 Roofline / 瓶颈分析

### 2.1 目标真机规格（`Ascend950DT_9592`）
- **核数与频率**：**36 AIC + 72 AIV** @ **1.65 GHz**（`core_count = 36` 参数化，未硬编码 32）
- **BF16 Cube 算力峰值**：单 AIC `13.5168 TFLOPS`（8192 FLOPs/cycle = 4096 MACs/cycle），36 核合计 **`486.6048 TFLOPS`**
- **存储带宽与片上容量**：HBM 96 GiB（标称 `4.0 TB/s`，实测写带宽 `~3.68 TB/s`）；单核 L1 = `512 KiB`（双 Bank 各 `256 KiB`），L0A = `64 KiB`，L0B = `64 KiB`，L0C = `256 KiB`，单 AIV UB = `256 KiB`

### 2.2 基准 Shape 理论计算量与搬运量

单 Query 计算量（5 个 GEMM，为前向的 2.5×）：$10 \times 64 \times 640 \times 512 = 209.72\text{ MFLOPs}$。

| 基准 Shape | 总 FLOPs | 100% Cube 峰值时间 | HBM 总搬运（`pre / main / post`） | `main` Core↔L2 总搬运 | `main` HBM OI | `main` Core↔L2 OI |
|---|---:|---:|---:|---:|---:|---:|
| **`4032×4096, topk=640`**（`36×112` 无尾波） | **845.57 GFLOPs** | **1737.7 us** | **1.37 GB** (`531.6 / 826.1 / 12.6 MB`) | **8.73 GB** | **1024 F/B** | **96.8 F/B** |
| **`4096×4096, topk=640`**（`36×113+28` 尾波 0.2%） | **858.99 GFLOPs** | **1765.3 us** | **1.39 GB** (`540.0 / 838.8 / 12.6 MB`) | **8.87 GB** | **1024 F/B** | **96.8 F/B** |

- **`main` Core↔L2 流量拆解（`4032×4096`，单 Query 2.166 MB）**：
  - `atomic scatter dKV_accum (FP32)`：**5.28 GB（60.5%）**
  - `gather KV (BF16)`：**2.64 GB（30.3%）**
  - `Q + dO + dQ + Indices + LSE/Delta`：**0.79 GB（9.2%）**
- **理论 Roofline 上限与真实硬件瓶颈**：
  - **Preprocess**：严格 **HBM 读带宽 bound**（531.6 MB，理论下界 ~144.5 us，v3 实测 202.5 us，`aiv_mte2_ratio = 95.5%`）。
  - **Postprocess**：12.6 MB 全在 L2 内，实测 8.2 us（占端到端 0.2%）。
  - **Main**：
    - 相对 HBM（OI = 1024 F/B ≫ 132 F/B 拐点）完全不是 HBM bound；端到端理论 MFU 天花板约 **89–92%**。
    - 但在实测中发现两道真实硬件约束：
      1. **Cube 小切块微架构效率 + 内循环 `bv` 同步**：v3 中 `aic_mac_ratio` 已达 **71.92%**（`2801 us` 活跃时间 vs `1738 us` 理论满载，64×64×128 与 K=64 的 64×128×64 切块加上 `dV/dK` 共用 `bv` 槽同步带来 ~1.6× 周期折损）。
      2. **FP32 稀疏原子写 `dKV_accum` 的 AIV MTE3 / L2 原子带宽墙**：v3 实测 `aiv_mte3_ratio = 54.7%`，即 MTE3 实际忙碌时长已达 **`3895 × 54.7% ≈ 2131 us`**（略高于 1738 us Cube 峰值），对应 72 AIV 稀疏 FP32 原子写等效吞吐 ~2.6 TB/s。
      3. **跨 Query 流水边界排空**：`for job` 外循环未打平，每 `nb = 10` 个 tile 排空一次 3 级 `T.Stage` 并串行 `dQ` drain 与下一 Query 的 `Q/dO` 预取（约占 15–18%）。

---

## 三、版本演进与关键优化（`v1 → v2 → v3`）

### 3.1 v1：纯 SIMD 正确性基线（`code/bwd_kernel.py`）
- 全部向量块改用纯 `T.SimdVF()`（规避了 Bisheng `-Ofast` 在 `__simt_vf__` 下对 `setcc f32<1e30>` 的 LLVM 选指崩溃）。
- 内层散写处局部声明 `T.assume_no_conflict(dKV_accum)` 解决 AIV0↔AIV1 MTE3 WAW 误报。
- 单版本 `T.Serial` 流水完成全量正确性验证；经 `npusim` 指令级 trace（`profiling/npusim_v1_bwd_main_trace.json`）诊断出 4 类气泡：L0 单缓冲停顿（97–185 ticks/MMAD）、`P/dS + InsertNd2Nz` 跨核等待（905 ticks）、KV gather 跨 tile 停顿（2663 ticks）、AIV 标量开销。

### 3.2 v2：多 Buffer + `gather2 / scatter2`（`code/bwd_kernel_v2.py`）
- `preprocess` / `postprocess` 升级双缓冲 `T.Pipelined(..., num_stages=2)`。
- `main` 引入 L0 多缓冲（`aq:2, bq:4, cdkv:2`）、L1/UB 多缓冲（`kv_l1_0/1:2, p_l1/ds_l1:2, kv_ub:2, chunk_fp32:2`）、提前 hoist `p_l1^T`/`ds_l1^T` 加载、`unit_flag_ctrl` + `T.set_mmad_direction`，以及内层 `for d` 与外层 `for k` 的 `T.Pipelined`。
- 实现成对 2-burst `gather2`（16 MTE2/AIV）及**无分支 0/1/2-burst `scatter2`**（`T.StridedTensor` + `T.assume(n_burst > 0)`，将 MTE3 原子写请求减半至 16 次/AIV 并解决 AutoSchedule `ResolveCore` 条件分支报错）。

### 3.3 v3：显式 3 级 `T.Stage` + 直出 Compact-NZ + L1 双 Bank（`code/bwd_kernel_v3.py`）
- **显式 3 级跨 KV tile 流水**（`for k in T.Pipelined(nb, num_stages=3, annotations={"enable_offset": True})`）：
  - `Stage 0`：`Indices` 加载 + pair-sort + `gather2` + SIMD `ND→NZ` → `kv_l1_0/1`（3 版本）
  - `Stage 1`：GEMM 1 (`S`) + GEMM 2 (`dP`) + 融合 `P/dS` → `pds_l1`（2 版本，合并 `P`/`dS` 保持跨核 flag ≤ 16）
  - `Stage 2`：GEMM 3 (`dQ`) + GEMM 4/5 (`dV+dK` 直写 `dkv_full`) + `scatter2`（藏在下一 tile 的 GEMM 1+2+3 之后）
- **原生 `L0C→UB` + 直出 Compact-NZ `P/dS`**：`s_ub/dp_ub` 采用 `make_ascend_l0c_layout`，`p_ub/ds_ub` 采用 `make_ascend_compact_nz_layout`，生成代码 `ascend_nd2nz_scatter` 降为 **0**，并消除每 tile 4 次 `chunk_fp32 → dkv_full` 拷贝 VF。
- **片上布局与排坑**：
  - L1 双 Bank 显式划分（Bank 0: `q_l1 / do_l1 / pds_l1` 160 KiB；Bank 1: `kv_l1_0/1` 192 KiB），5 个 GEMM 零 bank 冲突；L0B 拆分 `bq:2`（Stage 1, 32 KiB）与 `bv:2`（Stage 2, 32 KiB）消除跨阶段 owner 冲突。
  - `dQ` drain 直接复用单版本 `dkv_full` 原地打包 BF16，AIV 总 UB 仅 **~187 KiB**（远低于 256 KiB 动态 UB 上限），同时彻底规避了 TileLang `InsertSync` 的一个编译器边界缺陷（独立局部子循环触发跨核 flag 复用时，`HasCompatibleFlagKind` 未区分 `AIC→AIV` 与 `AIV→AIC` 方向导致反向 flag 0 遗漏初始 `arrive` 而死锁）。

### 3.4 Harness 长程闭环迭代与代表性版本（`v4a → v17`）

在 `v3`（`3895.34 us`, `44.61%` MFU）基础上，借助远程 950DT 上的 Operator Harness 进行了双路线系统探索（避免陷入 `BLOCK_N=64` 局部最优），共完成 `v4a`、`v5`、`v7`、`v8`、`v9`、`v10`、`v11`、`v12`、`v13`、`v14`、`v15`、`v16`、`v17` 以及去 `scatter2` 消融诊断，并精选 **3 个最具未来演进价值的代表性版本** 保留在代码库中：

1. **`v14`（当前全局最优，`code/bwd_kernel_v14.py`，`BLOCK_N=64` 展平深度流水）**：
   - 拆分 L0B 为 `bv_do:1 / bv_q:1` 消除 `dV → dK` 同步气泡；
   - 跨 Query 展平为单一 `T.Pipelined(n_steps, num_stages=3, enable_offset=True)` 连续流水（1120 步），用无分支 `n_qrows = if(k < 2, 64, 0)` + `T.assume(n_qrows > 0)` 避免重复加载 `Q/dO`；
   - 独立 16 KiB `dq_ub (32, 128)` 解耦 `dQ` drain 与 `scatter2`（消除 `v12` 中 `InsertSync` 将 `scatter2` 等待提升到每步循环的伪依赖）；
   - **`dkv_full: 2` 双缓冲**（128 KiB，总 UB ~233 KiB ≤ 250 KiB）解耦 `dV+dK` `PIPE_FIX` 与上一步 `scatter2`，将 `aic_mac_time` 从 `2935 us` 压降至 **`2081 us`**，主基准降至 **`3645.10 us`（`47.67%` MFU，相对 `v3` `-250.24 us` / `1.069×`）**。
   - 适合作为后续 **分组私有 `dKV` 缓冲 + `postprocess` 归约**、**连续 Query 分块打散 L2 热点** 或 **BF16 原子加** 优化的主基线。
2. **`v17`（最优 `BLOCK_N=128` 3-stage 展平版，`code/bwd_kernel_v17.py`）**：
   - 将 KV tile 加宽至 `BLOCK_N=128`（560 步），同时保持完整 3-stage 跨 Query 展平流水，并利用 `kv_ub (16,512) / nz_ub (17,512)` 省出 32 KiB UB 保持总 UB ~225 KiB；
   - 为在 512 KiB L1 内放下 `kv_l1_0/1:3`（384 KiB）+ `pds_l1:2`（64 KiB），将 `Q/dO` 改为从 L2 按 `(64, 128)` 宽块流式载入（`qdo1_l1:2` 32 KiB + `do2_l1:1` 16 KiB + `q2_l1:1` 16 KiB = 64 KiB，精确 512 KiB L1）；
   - 实测 **`3872.36 us`**（已优于 `v3`），且 **`aic_fixpipe = 1640 us`、`aiv_mte3 = 2000 us`**（原子写比 `v14` 快 `582 us`），唯一短板是 L2 流式重读 `Q/dO` 带来的 `aic_mte2 = 2126 us`。适合作为未来更大 L1 硬件或更优 `Q/dO` 复用策略的 `BLOCK_N=128` 基线。
3. **`v8`（`BLOCK_N=128` 2-stage `Q/dO` 常驻 L1 参考版，`code/bwd_kernel_v8.py`）**：
   - `Q/dO` 常驻 L1（`aic_mte2` 仅 `303 us`），完整 L0A/L0B/L0C 双缓冲（`aq:2, bq:2, bv:2, cdkv:2`），实测 **`aic_mac = 2180 us`、`aiv_mte3 = 1480 us`**（全部版本最低原子写耗时），仅因 2-stage 内 `gather2 → S+dP → P/dS` 串行及 5-tile/Q 排空导致总耗时 `4519.24 us`。保留作为 `BLOCK_N=128` 全片上 L0/L0C 双缓冲参考。

---

## 四、真机实测结果（36 核 `Ascend950DT_9592`）

### 4.1 全量精度验证（最优版 `v14`，对比 NPU FP32 分块参考解，FP64 `cos_diff`）

| 测试规格 | `dQ` (`max_abs` / `rel_max` / `cos_diff`) | `dKV` (`max_abs` / `rel_max` / `cos_diff`) | `dSink` (`max_abs` / `rel_max` / `cos_diff`) | 结论 |
|---|---|---|---|---|
| **非整除 + 全无效行（`sink=True`）**：`nq=65, nk=95, topk=128` | `0.01075` / `0.00320` / `3.06e-6` | `0.09145` / `0.00304` / `2.76e-6` | `0.03653` / `0.00216` / `1.70e-6` | **PASS** |
| **非整除（`sink=False`）**：`nq=65, nk=95, topk=128` | `0.01441` / `0.00345` / `3.22e-6` | `0.10266` / `0.00258` / `2.73e-6` | — | **PASS** |
| **36 核整除大 shape**：`nq=4032, nk=4096, topk=640, sink=True` | `0.01280` / `0.00436` / `2.49e-6` | `0.10409` / `0.00328` / `2.79e-6` | `0.02945` / `0.00140` / `7.97e-7` | **PASS** |
| **标准完整 shape（含尾波）**：`nq=4096, nk=4096, topk=640, sink=True` | `0.01369` / `0.00398` / `2.49e-6` | `0.10021` / `0.00323` / `2.79e-6` | `0.02488` / `0.00106` / `7.86e-7` | **PASS** |

> 注：`v8`、`v17` 亦均通过 `65/95/128 (all_invalid)` 与 `4032/4096/640` 双 shape 精度门禁（`cos_diff <= 3.06e-6`）。

### 4.2 性能与流水线利用率对比（`torch_npu.profiler Level1 + PipeUtilization + 1GB L2 flush`）

**主基准 `nq=4032, nk=4096, topk=640, core_count=36`（845.57 GFLOPs，峰值 486.60 TFLOPS）：**

| 版本 | 核心设计 | `pre` (us) | `main` (us) | `post` (us) | `e2e` (us) | `main` TFLOPS | `main` MFU | `aic_mac` (`us`) | `aic_fix` (`us`) | `aic_mte2` (`us`) | `aiv_vec` (`us`) | `aiv_mte3` (`us`) | 相对 v1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v1** | 纯 SIMD 基线 | 242.46 | 7690.50 | 8.55 | 7941.51 | 109.95 | 22.60% | 38.4% (2953) | 8.8% (677) | 2.6% (200) | 14.0% (1077) | 16.0% (1230) | 1.00× |
| **v2** | 多 buffer + `gather2/scatter2` | 203.12 | 5329.11 | 8.32 | 5540.55 | 158.67 | 32.61% | 43.0% (2292) | 65.7% (3501) | 4.8% (256) | 19.1% (1018) | 25.0% (1332) | 1.44× |
| **v3** | `BN=64`, 3-stage, 直出 NZ, 双 Bank | 202.55 | 3895.34 | 8.18 | 4106.08 | 217.07 | 44.61% | 71.9% (2801) | 58.4% (2275) | 9.7% (378) | 21.4% (834) | 54.7% (2131) | 1.97× |
| `v4a` | `v3` + 拆分 `bv_do/bv_q` | 202.4 | 3782.27 | 8.0 | 3992.68 | 223.56 | 45.94% | 77.6% (2933) | 52.3% (1979) | 10.7% (405) | 22.0% (832) | 61.1% (2312) | 2.03× |
| **`v8`** | `BN=128`, 2-stage, 全 L0/L0C 双缓冲 | 203.2 | 4519.24 | 8.3 | 4730.71 | 187.11 | 38.45% | 48.2% (2180) | 47.0% (2122) | 6.7% (303) | 34.1% (1543) | 32.7% (**1480**) | 1.70× |
| `v13` | `v4a` 展平 + 独立 `dq_ub`, `dkv:1` | 202.5 | 3705.48 | 8.1 | 3916.10 | 228.19 | 46.90% | 79.2% (2935) | 59.2% (2194) | 14.8% (548) | 23.3% (862) | 66.4% (2459) | 2.08× |
| **`v14` (最优)** | **`v13` + `dq_ub(32,128)` + `dkv_full:2`** | **202.50** | **3645.10** | **8.53** | **3856.13** | **231.97** | **47.67%** | **57.1% (2081)** | **88.9% (3242)** | **15.5% (566)** | **20.3% (740)** | **70.8% (2582)** | **2.11×** |
| `v16` | `v14` + 全 UB 双缓冲 (`s/dp/p/ds/dkv:2`) | 203.07 | 3680.11 | 8.24 | 3891.42 | 229.77 | 47.22% | 56.7% (2085) | 90.6% (3333) | 14.9% (548) | 20.7% (762) | 68.7% (2527) | 2.09× |
| **`v17`** | **`BN=128`, 3-stage 展平 + `(64,128)` Q/dO** | **203.00** | **3872.36** | **8.31** | **4083.68** | **218.36** | **44.87%** | **71.2% (2757)** | **42.4% (1640)** | **54.9% (2126)** | **37.5% (1453)** | **51.7% (2000)** | **1.99×** |
| `diag: v14 去 scatter2` | 诊断消融：仅移除 `dKV` 原子写 GM | — | **2596.63** | — | — | — | — | 79.1% (**2054**) | 96.1% (**2494**) | 10.1% (263) | 28.2% (733) | 14.5% (376) | — |

**标准完整 shape `nq=4096, nk=4096, topk=640, core_count=36`（858.99 GFLOPs，最优版 `v14`）：**
- `pre` = **205.48 us**（`aiv_mte2 = 95.4%`）
- `main` = **3700.05 us**（**232.16 TFLOPS / 47.71% MFU**，`aic_mac = 57.14%`，`aic_fixpipe = 88.92%`，`aiv_mte3 = 70.86%`，`cube_utilization = 98.87%`）
- `post` = **8.22 us**（`aiv_mte2 = 77.5%`）
- `e2e` = **3913.75 us**（**219.48 TFLOPS / 45.10% MFU**，相比 `v3` `4166.63 us` 下降 **`252.88 us`**）

### 4.3 真实硬件瓶颈定论与后续优化方向

1. **去 `scatter2` 消融实证的硬件墙**：
   - 即便完全移除 `dKV` FP32 稀疏原子写 GM，`v14` 在 `BLOCK_N=64` 下也已卡在 **`2596.63 us`**（`aic_mac = 2054 us`, **`aic_fixpipe = 96.06% / 2494 us`**，包含 `unit_flag_ctrl=3` 等待与 `dst_stride=512` 写 UB）。
   - 加入 `scatter2`（5.28 GB FP32 稀疏原子写）后，端到端额外增加 **`+1048.47 us`（`2596.6 → 3645.1 us`）**：不仅 `aiv_mte3` 暴增 `+2206 us`（`376 → 2582 us`），L2/UB 端口争用还将 `aiv_mte2` 拖慢 **1.86×**、`aic_mte2` 拖慢 **2.15×**、`aic_fixpipe` 拖慢 **1.30×**（`2494 → 3242 us`）。在 `v16` 中把全部 UB 缓冲双缓冲后 `aic_fixpipe` 仍饱和在 `90.56%`。
2. **为什么 KV-outer（按 KV 切分核）对 V4.1 稀疏 MQA 不可行**：
   - V4.1 为极端 MQA（`H_q=64, H_kv=1`），单 Query 的 `Q+dO` 为 `128 KiB`，而单 KV 仅 `1 KiB`（128:1 不对称）。按 KV 切分不仅会使 `Q+dO` HBM 读流量放大 `10×` 以上，且非结构化 top-k 会造成严重的块内有效率崩塌与热点 KV 负载不均，因此**按 Query 切分仍是主并行轴**。
3. **未来最值得推进的片内优化方向（基于 `v14`）**：
   - **分组私有 `dKV_accum[G, nk, 512]` + `postprocess` 归约**（如 `G=4` 仅 32 MiB 驻留 L2，按 `bx % G` 将相邻 Query 打散到不同副本，把同一 KV 行的 L2 原子热点冲突降为 `1/G`，`postprocess` 归约仅增加 ~25–40 us）；
   - **连续 Query 分块调度**（`qidx = bx * jobs_per_core + job` 替代跨步 `bx + job * 36`，零额外显存打散相邻 Query 的重叠索引热点）；
   - 若精度契约允许，探索 **BF16 原子加**（将 scatter 流量与 `dkv_full` UB 减半）。

---

## 五、本目录文件索引

```text
bwd_report/
├── README.md                              # 本总结报告
├── code/                                  # TileLang 源码快照（同步自 examples/ascend/flashmla/）
│   ├── bwd.py                             # 独立编译 / FP64 精度校验 / 真机 Profiler 入口
│   ├── bwd_kernel.py                      # v1 正确性基线
│   ├── bwd_kernel_v2.py                   # v2 多 buffer + gather2/scatter2
│   ├── bwd_kernel_v3.py                   # v3（3 级 T.Stage + 直出 NZ + L1 双 Bank）
│   ├── bwd_kernel_v8.py                   # v8（BLOCK_N=128 2-stage 全 L0/L0C 双缓冲参考版）
│   ├── bwd_kernel_v14.py                  # v14（当前全局最优：BLOCK_N=64 展平 + dkv_full:2）
│   └── bwd_kernel_v17.py                  # v17（最优 BLOCK_N=128 3-stage 展平 + L2 流式 Q/dO）
├── ascendc/                               # 编译生成的 AscendC 源码 (.asc)
│   ├── v1_4096/{bwd_preprocess,bwd_main,bwd_postprocess}.asc
│   ├── v2_4032/{v2_bwd_preprocess,v2_bwd_main,v2_bwd_postprocess}.asc
│   ├── v3_4032/{v3_bwd_preprocess,v3_bwd_main,v3_bwd_postprocess}.asc
│   ├── v3_65/{v3_bwd_preprocess,v3_bwd_main,v3_bwd_postprocess}.asc
│   ├── v14_4032/{v14_bwd_preprocess,v14_bwd_main,v14_bwd_postprocess}.asc
│   └── v17_4032/{v17_bwd_preprocess,v17_bwd_main,v17_bwd_postprocess}.asc
└── profiling/                             # 真机 950DT msprof 流水数据 + npusim 指令级 Trace
    ├── npusim_v1_bwd_main_trace.json      # v1 CAmodel 指令级流水（Chrome / Perfetto 可视化）
    ├── v1_4032_c36/{pre,main,post}_{kernel_details.csv,trace_view.json}
    ├── v2_4032_c36/{pre,main,post}_{kernel_details.csv,trace_view.json}
    ├── v3_4032_c36/{pre,main,post}_{kernel_details.csv,trace_view.json}
    ├── v3_4096_c36/{pre,main,post}_{kernel_details.csv,trace_view.json}
    ├── v14_4032_c36/{pre,main,post}_kernel_details.csv
    ├── v14_4096_c36/{pre,main,post}_kernel_details.csv
    └── v17_4032_c36/{pre,main,post}_kernel_details.csv
```

### 复现命令（950DT 远程环境）

```bash
source /home/sym/zsm/runs/flashmla-dt/env.sh
cd /home/sym/zsm/tilelang

# 1. CPU 预编译 + 导出 AscendC（无需 NPU 锁，可选 --version v3/v8/v14/v17）
python -u examples/ascend/flashmla/bwd.py --mode compile --version v14 --nq 4032 --nk 4096 --topk 640 --core-count 36 --dump-dir /tmp/asc_v14

# 2. 真机全量精度校验 + 性能采集（必须通过 npu-guard）
npu-guard run --owner "$USER/flashmla-bwd-v14" --wait 3600 --timeout 300 -- bash -c '
  source /home/sym/zsm/runs/flashmla-dt/env.sh && cd /home/sym/zsm/tilelang
  python -u examples/ascend/flashmla/bwd.py --mode verify --version v14 --nq 4032 --nk 4096 --topk 640 --core-count 36 --ref-device npu
  python -u examples/ascend/flashmla/bwd.py --mode bench  --version v14 --nq 4032 --nk 4096 --topk 640 --core-count 36 --prof-dir /tmp/prof_v14
'
```
