# AutoSchedule 的两类时间与约束

可以把流水调度的两类主要约束理解为：**存储中的数据何时可以被访问或覆盖**，以及**硬件资源何时可以接收下一个任务**。代码中不把两种时间都叫 II：

| 含义 | 代码中的表示 | 用途 |
| --- | --- | --- |
| 任务执行延迟，以及据此确定的 buffer 依赖等待时间 | `latencies[i]`；依赖边中的 `latency`，下文记为 `wait` | 保证数据就绪、读写完成后才进行后续访问或覆盖 |
| 任务在模型中的硬件资源占用间隔 | `iis[i]`，下文记为 `II_i` | 保证有资源冲突的任务占用区间不重叠 |

这里的 buffer 等待时间不宜直接称为“buffer 的 II”：它是**依赖边要求的等待时间**；一块 buffer 从生产到最后一次使用的完整生命周期还可能跨越多个任务。

## 共同的时间表示

设循环每隔 `P` 个周期启动一轮。任务的模型开始时间为：

\[
S_i=k_iP+r_i,\qquad 0\le r_i<P
\]

`k_i` 表示任务跨了多少个循环周期，`r_i` 表示它在周期内的启动位置。第 `n` 轮的该任务在 `S_i+nP` 开始。

实现：[开始时间构造](/C:/Users/zhang/dev/tilelang-ascend-cce/tilelang/ascend/transform/z3_scheduler.py:428)，变量为 `k_vars`、`r_vars` 和 `start_vars`。

## 1. Buffer 读写依赖：约束完整开始时间

`data_deps` 的一条边 `(u, v, distance, latency)` 表示：第 `n+d` 轮的任务 `v`，必须等待第 `n` 轮的任务 `u` 开始后至少 `wait` 个周期：

\[
\boxed{S_v+dP\ge S_u+wait}
\]

即代码中的 `start_vars[v] - start_vars[u] >= latency - P * distance`。

- `d=0`：同一轮内的依赖。
- `d>0`：跨轮依赖。
- 对受自动 buffer 份数控制的覆盖依赖，用份数 `b` 作为距离：`S_v+bP >= S_u+wait`。增加份数，就能让同一槽位更晚被重新使用。

代码用负的 `distance` 编码“引用哪个份数变量”，**不表示实际依赖距离为负数**。并非每条 buffer 依赖都随份数变化。

实现：[数据依赖约束](/C:/Users/zhang/dev/tilelang-ascend-cce/tilelang/ascend/transform/z3_scheduler.py:460)，自动份数保存在 `buffer_vars`。

## 2. 硬件资源互斥：约束周期内位置

`resource_deps` 记录有硬件资源冲突的任务对。若求解器选择任务 `i` 在周期内先于任务 `j`，则同时满足：

\[
\boxed{r_j-r_i\ge II_i}
\]
\[
\boxed{r_i+P-r_j\ge II_j}
\]

第一条保证本周期内 `i` 与 `j` 的资源占用不重叠；第二条保证 `j` 与下一个周期的 `i` 也不重叠。若选择 `j` 在前，则交换两者，使用另一组对称不等式。

这里比较的是周期位置 `r_i`，而不是完整开始时间 `S_i`：任务即使相差几个调度阶段，稳态下不同轮次仍会使用同一硬件资源。**这组约束不直接依赖 buffer 份数。**

实现：[资源互斥约束](/C:/Users/zhang/dev/tilelang-ascend-cce/tilelang/ascend/transform/z3_scheduler.py:474)，`iis` 保存间隔，`o_ij` 选择先后顺序。

例如两个互斥任务的 `II` 分别为 10 和 20，将上面两式相加便得到 `P >= 30`；增加 buffer 份数不能消除这条硬件下界。

因此，两类约束分别保证**数据访问与覆盖安全**和**硬件资源占用不重叠**，求解出的时间安排必须同时满足它们。它们是两类主要约束，模型还包含容量、阶段及其他调度约束。
