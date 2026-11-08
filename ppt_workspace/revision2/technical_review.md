# 第二轮 PPT 技术口径核对

核对日期：2026-11-08。依据本仓库当前源码和官方产品白皮书。本文为制作与口播核对材料，不直接放进投影片。

## 1. 硬件规格与前段等时示例

采用 **Ascend 950PR，32 个 Cube Core、64 个 Vector Core，FP16 Cube 峰值 432 TFLOPS**。

官方《昇腾 950 NPU 架构白皮书》表 3-1（PDF 第13—14页）区分了：
- 32核档：Cube FP16/BF16 432 TFLOPS；Vector 54 TFLOPS；合计486 TFLOPS。
- 28核档：Cube 378 TFLOPS；Vector47 TFLOPS；合计425 TFLOPS。

**不要将486或425当成32核Cube峰值。**

来源：[昇腾950PR官方产品页](https://www.hiascend.com/hardware/processor)，该页链接的[官方架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)。

前段展示每批8个等时任务：搬入 → QK → 搬出 → softmax → 回传 → PV → 搬出 → 累加。每批Cube承担2项，因此忙碌时间比例=2/8=25%。若32核同样调度，且Cube工作期间按峰值计，则有效Cube算力=432×25%=108 TFLOPS。页面应写“等时示例，Cube工作时按峰值计”。这不是真实完整FlashAttention耗时分配，也与末段23.35%→90.77%实测数据独立。

末段数据按用户提供的实测值写 **Cube MFU：23.35% → 90.77%**，不得再写成98.77%、理论值或自行还原出未提供的测量配置。出处可只在备注写“讲者提供的实测结果”。

## 2. 变量与时间

|符号|讲解口径|
|---|---|
|II|整个流水循环相邻逻辑迭代的启动间隔，待搜索|
|k_i|任务i相对逻辑迭代起点的调度阶段，非负整数|
|r_i|任务i在一个II周期内的发射位置，0≤r_i<II|
|b|某个buffer的版本/份数，正整数；严格来说按buffer求解，任务之间可共享|
|L_i|本页基本任务示例中，任务i的数据访问完成等待时间；用于结果就绪/读取结束约束|
|H_i|同一硬件资源再次接收冲突任务前的占用间隔，源码iis[i]|
|S_i|第0轮任务i的模型开始时间，S_i=k_i·II+r_i；第n轮为S_i+n·II|

**L_i不代表从最初写入到最后消费者读取结束的完整buffer生命周期。** 完整生命周期跨越生产者与消费者，不能只看生产任务的latency。源码一般依赖使用按边推导的wait，简单任务场景下可写为L_u。汇总任务的边等待存在偏移，不必在主讲中展开。

## 3. 两类数据约束

生产者u到消费者v：

`S_v + d·II ≥ S_u + L_u`

d=0表示同轮；d>0表示跨轮。首次解释时明确d，不让它突然出现。

同一份buffer跨轮复用：

`S_write + b·II ≥ S_lastread + L_lastread`

即第n+b轮对该槽的下一次写入，必须等第n轮旧数据的最后一次读取完成。画“写入 → 保留 → 最后读取完成 → 下一次写入”的直线时间轴；两份buffer时用 n mod 2 解释轮换。**不能写成仅需 S_write+bII≥S_write+生产延迟**，否则会漏掉消费者仍在读取的时间。

## 4. 硬件使用约束：两个if分支，每个分支均含两条不等式

if i 在周期内先于 j：

`r_j − r_i ≥ H_i`

`r_i + II − r_j ≥ H_j`

if j 在周期内先于 i：

`r_i − r_j ≥ H_j`

`r_j + II − r_i ≥ H_i`

后一条保护周期交界，不能删除。可使用两行线性时间轴，无需圆环。比较r而非S，因为不同阶段、不同轮次在稳态下仍争用同一设备。

这里的“不能并行”是模型中的**冲突资源占用区间**互斥，不意味着设备内部完全不能流水化；H_i可能小于L_i。

## 5. 建模总结与求解

页面包含：S_i=k_iII+r_i、变量域、数据依赖、buffer复用、资源两个if，以及片上容量 Σ(b×单份buffer大小)≤容量。若页空间不够，容量可做一行，不能让观众以为buffer可无限增加。

配色建议：已知量L/H/d/容量用灰蓝；未知量k/r/b及顺序用红；II用橙。不要将S_i误标成独立待求变量：它由k/r/II派生。

直接min II中存在k_i×II、b×II等未知数乘积。固定候选II后，问题成为带整数、布尔顺序条件的线性约束可满足性检查（SMT），不要称为纯线性规划LP。

源码二分可行II：`ii_lower=max(iis)`、`ii_upper=max(sum(latencies),sum(iis))+1`，候选II下Z3检查。搜索资源下界图可作宏观原理，但不要说源码直接以“最忙设备总占用”初始化。没有隐藏页后这点放讲稿备用说明即可。

实际源码设求解预算；unknown会沿不可行分支处理，因此不承诺所有工程配置下总能证明全局数学最小值。投影片写“搜索更小的可行II”自然准确。

## 6. 性能目标的桥接

固定工作量F与峰值P时，MFU=F/(T·P)，所以T缩短对应MFU提高。周期性调度下：

`T = (N−1)·II + 末轮完成跨度`

末轮跨度从末轮起点算到其最晚任务完成。长循环中(N−1)·II项主导；不要把末轮跨度说成所有排班下绝对常数。保留首尾开销口播即可，图上着重相邻起点间距。

## 7. 源码依据

- `tilelang/ascend/language/schedule_hint.py:65`：Task latency与ii参数语义。
- `tilelang/ascend/transform/z3_scheduler.py:406`：II二分初始界。
- `tilelang/ascend/transform/z3_scheduler.py:428`：k/r/start构造。
- `tilelang/ascend/transform/z3_scheduler.py:445`：buffer份数和容量约束。
- `tilelang/ascend/transform/z3_scheduler.py:460`：数据依赖与buffer版本距离。
- `tilelang/ascend/transform/z3_scheduler.py:474`：资源互斥四个Implication。
- `src/ascend/transform/auto_schedule.cc:351`：依赖等待按实际任务对及时间偏移推导。
- `src/ascend/transform/auto_schedule.cc:1601`：循环整体延迟计算。
