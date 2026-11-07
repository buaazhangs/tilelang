做一个PPT，并提供适当的发言稿
PPT汇报时间：35-40min
主要内容：
介绍当前仓库，tilelang支撑昇腾算子高性能运行的方式，核心即autoschedule 的模调度
面对对象：
大学学生和教授，以算法研究为主，做infra的了解GPU 为主
思维导图：
各个模块按照因果关系，层层递进
整体上，我打算用：C:\Users\zhang\dev\tilelang-ascend-cce\工人.md 这里提及的工人工作比喻，串起整个PPT
然后，我打算这样设计（讨论这部分，包括每部分的细节和页数，当然大模块也可以改，比如不分成下面三部分，可以讨论）：
一 硬件利用率：算子编写的核心目的之一，介绍TFLOPS MFU等基本概念，让听众意识到这部分是算子的一个优化关键
二 关键区别：GPU和NPU执行逻辑上的根本不同：（可参考C:\Users\zhang\dev\tilelang-ascend-cce\GPU vs NPU.md）-> 引出昇腾为什么需要静态编排各个任务，及下一章节
三 tilelang 编译器的任务：
A  
B 建模方案，基于buffer占用 和 设备占用的两组不等式，即C:\Users\zhang\dev\tilelang-ascend-cce\autoschedule_time_constraints.md
C 求解方案，过渡到
并基于思维导图分析：
①每页提炼一个关键词
②自然过渡（页面内的模块-模块，还有页面-页面之间的过渡，清晰的关键连接语）
先不开始做，先讨论下