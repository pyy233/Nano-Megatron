# Megatron-LM 与 Nano-Megatron 架构对比

> 调研对象：`/home/qep/WorkSpace/Megatron-LM`
>
> 源码版本：`main@779c5b748`
>
> 目的：不是评价 Megatron-LM 的代码风格，而是判断哪些生产设计应当保留，哪些复杂度不适合教学型 mini 实现。

## 1. 结论先行

Megatron-Core 最值得借鉴的不是某个具体类，而是四个成熟抽象：

1. 用正交 rank generator 描述复合并行组。
2. 模型组件通过 spec/builder 替换，而不是把 fused kernel、MoE 等分支写死在 GPTModel。
3. dense 参数和 expert 参数使用不同的 replica group、buffer 和梯度缩放。
4. checkpoint 先描述逻辑全局张量及其 shard metadata，再选择物理存储策略。

Nano-Megatron 不应照搬的部分是：

- 模块级 `parallel_state` 与显式 `ProcessGroupCollection` 两套依赖路径并存。
- dense/expert 使用两套 rank generator，且单个 generator 不允许 CP、EP 同时大于 1。
- 单个超大配置对象承载模型、训练、kernel、MoE、推理和 offload 的所有开关。
- 为兼容大量模型和历史行为而形成的长 schedule、optimizer 与 fallback 分支。

因此修订后的 Nano-Megatron 采用：

```text
单一五维 topology
  -> 声明式 GroupPlan
  -> immutable ParallelContext
  -> 显式 ParameterDomain
  -> typed component factory
  -> 小型 schedule/strategy 对象
  -> ShardMetadata + PyTorch DCP
```

## 2. Megatron-LM 的整体分层

从源码看，Megatron-LM 实际包含两层产品：

```text
megatron/training
  参数解析、初始化、数据、pretrain/train_step、日志、平台集成
        │
        ▼
megatron/core
  模型组件、并行算法、pipeline schedule、DDP/optimizer、checkpoint
        │
        ├── Apex / Triton / fused kernels
        └── torch.distributed / NCCL / Gloo / UCC
```

`megatron/core/README.md` 将 Core 定位为“用于构建自定义训练框架的 production-ready library”。这个定位解释了它为什么同时需要低层可组合组件、完整训练入口、多个硬件后端和大量兼容配置。

Nano-Megatron 第一阶段只对应其中较窄的一部分：一个 GPT 训练引擎和可读的并行参考实现，不承担完整平台能力。

### 2.1 核心源码导航

| 源码 | 主要职责 |
|---|---|
| `megatron/core/parallel_state.py:446` | RankGenerator、全局 group 初始化与 getter |
| `megatron/core/process_groups_config.py:27` | 显式 `ProcessGroupCollection` 与全局兼容桥 |
| `megatron/core/models/gpt/gpt_model.py:95` | GPT embedding、TransformerBlock、LM head、PP pre/post process |
| `megatron/core/transformer/spec_utils.py:12` | `ModuleSpec` 动态组件描述 |
| `megatron/core/transformer/transformer_layer.py:305` | submodule spec 装配、attention/MLP/MoE/recompute |
| `megatron/core/pipeline_parallel/schedules.py:48` | schedule 选择、无 PP/1F1B/interleaved forward-backward |
| `megatron/core/distributed/distributed_data_parallel.py:25` | dense/expert grad buffer 与 overlap |
| `megatron/core/optimizer/distrib_optimizer.py:107` | optimizer state/gradient shard 与 param all-gather |
| `megatron/core/transformer/moe/moe_layer.py:164` | router、dispatcher、local/shared experts 装配 |
| `megatron/core/transformer/moe/token_dispatcher.py:56` | all-gather/all-to-all/flex token dispatch |
| `megatron/core/dist_checkpointing/mapping.py:52` | `ShardedTensor` 逻辑全局布局 metadata |
| `examples/run_simple_mcore_train_loop.py` | 最小 Core 用户调用链 |

## 3. 并行状态与进程组

### 3.1 Megatron 的优势

`megatron/core/parallel_state.py:446` 的 `RankGenerator` 已把 TP、PP、CP、EP、DP 建模为可排序轴，并通过 mask 生成 `tp-dp` 等任意复合 rank group。这比为每个组合手写 rank 算法可靠得多。

默认 rank order 为 `tp-cp-ep-dp-pp`，允许根据节点拓扑调整相邻 rank。对大规模集群而言，rank order 会直接影响 TP/EP 等高频通信是否落在 NVLink/NVSwitch 内，是必须保留的能力。

`megatron/core/process_groups_config.py:27` 的 `ProcessGroupCollection` 把 TP、PP、CP、EP、TP×CP、TP×EP、DP×CP、expert-DP、embedding、optimizer instance 等 group 集中到可传递对象中。GPT、TransformerLayer、DDP、optimizer 和 gradient finalization 已逐渐显式接收它。

这证明用户希望的“parallel state 是构造函数对象”不仅适合教学，也符合 Megatron-Core 当前的演进方向。仓库自身的 `AGENTS.md` 也明确要求 core 新代码优先传入 `ProcessGroupCollection` 或显式 `ProcessGroup`，避免新增全局 getter。

### 3.2 Megatron 的代价

迁移尚未完成：

- `ProcessGroupCollection.use_mpu_process_groups()` 从全局 `parallel_state` 抽取 group。
- `TransformerLayer`、`TransformerBlock`、MoELayer、optimizer 等仍在参数为 `None` 时回退全局状态。
- checkpoint、training 和部分工具仍直接读取全局 getter。

这让组件既可以显式注入，也可以隐式工作，兼容性很好，但测试时很难保证一个对象真正只依赖传入的 group。

此外，`ProcessGroupCollection` 使用不断增加的可选字段表达新组合。当前文件约 718 行，已经包含多个同 ranks、不同用途的 communicator。字段式 API 易懂，但随着 hierarchical CP、多 optimizer instance、多模块 pipeline 等能力增加，会出现明显的组合膨胀。

### 3.3 CP 与 EP 的双 topology

`RankGenerator` 当前包含以下限制：

```python
assert ep == 1 or cp == 1
```

`initialize_model_parallel()` 随后构造：

- dense decoder generator：`ep=1`，包含 TP/PP/CP/DP。
- expert generator：`cp=1`，包含 expert-TP/PP/EP/expert-DP。

这个方案能正确得到 dense DP×CP 和 expert-DP group，也保持 PP group 对齐，但 CP、EP 的关系被分别折叠进两套 DP size 中。它适合兼容既有 Megatron rank 语义，却不符合“EP 是配置中的独立一等轴”这一目标。

### 3.4 Nano 的修订

Nano 使用一个五维 topology：

```text
world = TP × PP × CP × EP × DP
```

然后通过 group algebra 派生：

```text
EP axis group       = EP
dense replica group = DP × EP × CP
expert replica group= DP × CP
batch replica group = TP × PP × CP（固定 DP、EP）
```

这样 EP 的坐标、size 和 rank order 始终可见，同时保留 Megatron 中正确的 dense/expert 梯度语义。

这里有一个容易误解但必须明确的事实：EP 虽然是独立拓扑轴，在 dense layer 中仍表现为额外的数据副本；否则所有 EP rank 会重复计算同一批 dense token，造成浪费。配置中的独立轴不意味着它在每类参数上都扮演相同角色。

## 4. 模型构造与组件替换

### 4.1 Megatron 的 ModuleSpec

`megatron/core/transformer/spec_utils.py` 定义 `ModuleSpec`，可以保存 module class/import path、params、submodules 和 metadata。

GPTModel 接收一个 transformer layer spec；TransformerLayer 再通过 submodule spec 构造：

- norm
- self/cross attention
- bias-dropout-add
- dense MLP 或 MoE
- local、fused、inference-optimized 等后端

`megatron/core/models/gpt/gpt_layer_specs.py` 中的大量 builder 证明这套设计有很强的模型变体承载能力。GPTModel 本身不需要知道每个 fused module 或 MoE expert 的具体类。

### 4.2 ModuleSpec 的成本

- params/kwargs 以字典在多层 builder 间传播，静态类型较弱。
- 构造失败往往在运行时才暴露。
- 为兼容 MLP/MoE 不同签名，TransformerLayer 中存在条件 rewrap 和 kwargs 转发逻辑。
- 阅读者必须同时跟踪 spec、submodule dataclass、builder 和最终 module。

这对支持数十种模型是合理成本，对 mini GPT 则过重。

### 4.3 Nano 的选择

Nano 保留“GPTLayer 不绑定具体组件”的思想，但使用 typed `GPTComponentFactory`：

```text
DenseGPTComponents
  ├── build_attention
  ├── build_mlp
  └── build_norm

MoEGPTComponents（第二阶段）
  └── 只替换目标 layer 的 build_mlp
```

优点是构造调用链清楚、IDE/type checker 能看到接口；缺点是不如 ModuleSpec 支持任意第三方动态模块。对教学项目，这是合适的交换。

## 5. Pipeline 与训练控制流

### 5.1 Megatron 的能力

`megatron/core/pipeline_parallel/schedules.py` 约 2497 行，包含：

- 无 pipeline 的 forward/backward。
- 非交错 1F1B。
- interleaved/virtual pipeline。
- warmup/steady/cooldown。
- output pseudo-deallocation。
- P2P overlap、batch communication。
- partial activation checkpoint。
- multimodule pipeline、MoE overlap、CUDA graph 等兼容路径。

函数式 schedule 可以把关键路径压得很紧，也便于做大量细粒度优化。Megatron 的生产吞吐很大程度依赖这些分支。

### 5.2 Megatron 的阅读成本

schedule 接收大量参数，并从 config、model wrapper、parallel state/P2P communicator 中读取行为。一次 microbatch 的状态分散在多个局部 list、queue 和 helper 中。

`examples/run_simple_mcore_train_loop.py` 即使只是两层 GPT，也需要用户手工完成 distributed 初始化、parallel state、seed、模型、DDP、forward step callback、schedule、gradient finalization、optimizer 和 sharded checkpoint。`megatron/training.pretrain()` 能隐藏这些细节，但又进入更大的全局 args/training framework。

### 5.3 Nano 的选择

Nano 把 schedule 做成小型状态对象：

- `GPipeSchedule` 是正确性 oracle。
- `OneForwardOneBackwardSchedule` 只做非交错调度。
- P2P 由单独 communicator 对象负责。
- `DataParallelStrategy.microbatch_context()` 处理是否同步梯度。

这牺牲 virtual pipeline 和高级 overlap，但一个学习者可以从 schedule 的 event trace 完整看到每个 microbatch 的 forward、send、recv、backward。

## 6. DDP、ZeRO 与参数域

### 6.1 Megatron 的关键正确性设计

`megatron/core/distributed/distributed_data_parallel.py` 会按以下 key 分组参数：

```text
(parameter dtype, gradient dtype, is_expert_parallel)
```

dense 参数 buffer 使用 DP×CP group，expert 参数 buffer 使用 expert-DP group。两类参数还需要不同的 pre-scaling，才能最终都按 dense data-parallel world size 得到正确梯度。

这是初稿最重要的遗漏：单个 `parallel.dp_mesh` 无法正确处理未来 MoE。

### 6.2 Megatron distributed optimizer

`megatron/core/optimizer/distrib_optimizer.py` 约 3100 行，包含：

- contiguous param/grad buffer。
- bucket owner range 与 optimizer shard。
- gradient reduce-scatter、parameter all-gather 及 overlap。
- FP32 main parameter 与多精度状态拷贝。
- 多 distributed optimizer instance。
- 多种 optimizer checkpoint/reshard 格式。
- FP8/FP4 等兼容路径。

它的优势是与 Megatron DDP、pipeline、checkpoint 和 fused kernel 深度协同；缺点是很难作为 ZeRO 原理的第一份阅读材料。

### 6.3 Nano 的选择

- ZeRO-1/2 保留较短的 flat bucket + sharded AdamW，用于展示状态、梯度分别在哪个阶段分片。
- ZeRO-3 使用 FSDP2 管理参数 materialization/hooks。
- 所有 bucket 和 FSDP unit 显式携带 `ParameterDomain`。
- dense 参数使用 `DENSE_REPLICA=DP×EP×CP`，expert 参数使用 `EXPERT_REPLICA=DP×CP`。

精度实现也更窄：FP16 params 只允许维护 FP32 master shard 的 ZeRO-1/2；DDP 和
ZeRO-3 拒绝 FP16 params。FP16 compute 没有 GradScaler/动态 loss scaling，低精度
vocab-parallel cross entropy 则固定在 FP32 中计算数值敏感部分。

优势是概念边界清楚；风险是同一模型中使用不同 FSDP2 mesh 的组合需要尽早做原型验证。第二阶段实现 MoE 前，应先用一个 toy dense/expert 双模块验证 forward、backward、checkpoint 和重计算。

## 7. TP、SP、CP 与优化 kernel

Megatron TP 层、autograd mapping、vocab parallel loss 已经成熟，生产优化路径还支持
多种 CP communication type、hierarchical CP 和 fused attention。

优点：

- 高性能路径完整。
- 优化栈能统一 fused attention、低精度、RNG 和 CP communicator。
- 大量模型和硬件组合经过测试。

缺点：

- CP 的部分算法细节位于外部扩展，单读 Megatron-Core 不容易看到 ring/online-softmax 的完整过程。
- local/reference 路径与最优路径能力不对称。
- 外部扩展版本与 GPU 能力成为行为矩阵的一部分。

Nano 只保留 PyTorch reference backend。它不会比 Megatron 的优化路径快，但更适合
学习通信语义和定位数值问题。

## 8. EP 与 MoE

Megatron MoE 已有较好的内部 protocol：Experts、Router、SharedExperts builder，以及 all-gather、all-to-all、flex/DeepEP 等 dispatcher。local expert 数量由 `num_experts / ep_size` 决定，expert tensor parallel group 也可以与 attention TP 不同。

生产优势：

- dispatcher 和 experts 可独立替换。
- 支持 shared expert、grouped GEMM、通信 overlap、router loss、capacity、token drop 等。
- 对大模型和新硬件后端扩展速度快。

教学成本：

- token dispatcher 文件本身约 1859 行。
- router、dispatcher、experts、shared expert、CUDA graph、quantization 之间存在大量组合。
- 一部分路径仍有全局 process-group fallback。

Nano 第二阶段只实现：

1. top-k router。
2. contiguous local experts。
3. 一个可读 all-to-all dispatcher。
4. aux loss 和 token count 统计。
5. dense/expert 参数域的 DDP/ZeRO/checkpoint。

DeepEP、shared expert overlap、capacity/drop token、expert tensor parallel 留到后续。

## 9. Activation recomputation 与 Offload

Megatron 已支持 selective/full recompute，以及按模块组进行 activation offload。细粒度 offload 子系统维护 pinned CPU pool、D2H/H2D stream、event、inflight 限制，并和 PP warmup、CUDA graph replay 协同。

这说明 offload 不是简单的 `tensor.cpu()`，而是独立的内存调度系统。Nano 应保留三层：

```text
OffloadPolicy     选择什么、何时 offload
OffloadHandler    pinned buffer、stream、event
Module boundary   checkpoint/saved-tensor/block hook
```

第一阶段只实现 block/saved-tensor 粒度和有限异步拷贝，不做 Megatron 的逐子模块 PP-aware offload。

## 10. Distributed checkpoint

Megatron `ShardedTensor` 保存：

- logical key。
- local/global shape。
- global offset。
- axis fragmentation。
- replica id。
- 可选 factory/transform。

save/load strategy 再决定 Torch distributed、fully-parallel 等物理执行方式。这种“逻辑布局先于存储后端”的分层很强，能够支持 topology 变化和 checkpoint reshard。

Nano 初稿只写“用 PyTorch DCP、只支持改变 DP degree”，扩展边界过窄。修订后增加 `ShardMetadata`/`ShardedState`，DCP 只是第一个 storage adapter。

Nano 不会第一阶段实现 Megatron 的完整转换能力，但不会把 checkpoint state dict 设计成只能理解当前 local tensor。

当前物理实现按策略分开：单进程非 ZeRO-3 优先使用 DCP；多进程 DDP/ZeRO-1/2
使用共享文件系统上的 rank-local shard，并支持 dense DP/EP replica reshard。
ZeRO-3 通过 PyTorch state-dict API 导出 FSDP2 canonical model/optimizer state，再由
DCP 保存；dense GPT 按 `(TP, PP)` coordinate 使用独立子目录。它要求 TP/PP/CP/EP
degree 固定，但支持 DP degree resize。

## 11. 配置架构

Megatron `TransformerConfig` 当前约 2893 行，覆盖模型结构、初始化、融合、recompute、FP8/FP4、MoE、CP、CUDA graph、inference、offload、Mamba/MLA 等。

优点：单个对象能把几乎所有能力传到深层 module，生产脚本配置完整。

缺点：

- 字段之间存在大量跨域约束。
- module 容易依赖与自身无关的配置。
- 学习者难以分辨稳定核心与实验功能。

Nano 坚持分块配置：

```text
ParallelConfig
GPTConfig
KernelConfig
DataParallelConfig
PipelineConfig
ActivationCheckpointConfig
OffloadConfig
CheckpointConfig
TrainingConfig
```

每个模块只接收所需 config 或只读 view；跨块校验集中在启动阶段。

## 12. 两套架构的优劣总结

| 维度 | Megatron-LM | 修订后的 Nano-Megatron |
|---|---|---|
| 性能上限 | 极高，通信 overlap 和 fused kernel 成熟 | 第一阶段明显较低 |
| 功能覆盖 | 大量模型、硬件、精度、训练/推理场景 | 只覆盖 GPT 训练核心路径 |
| 并行拓扑 | 成熟 rank generator，但 dense/expert 双拓扑 | 单一可配置五轴 topology |
| 依赖注入 | 正在迁移；显式 collection 与全局 fallback 并存 | 从第一天强制显式 context/group |
| 新 group 扩展 | 增加 collection 字段和初始化路径 | 新增声明式 GroupSpec |
| 模型扩展 | ModuleSpec 极灵活 | typed factory 更窄但更清楚 |
| 参数域 | 已在 DDP/optimizer 中成熟处理 | 显式 domain registry，已覆盖 dense EP/CP replica、DDP/ZeRO、global norm 与 checkpoint |
| PP | 功能/优化丰富 | GPipe + 非交错 1F1B，易追踪 |
| ZeRO/优化器 | 深度定制、性能高、代码复杂 | ZeRO-1/2 教学实现，ZeRO-3 借 FSDP2 |
| Checkpoint | 完整 sharded metadata 和 strategy | DDP/ZeRO-1/2 rank-local shard；ZeRO-3 canonical DCP，固定 TP/PP/CP/EP、支持 DP resize |
| Precision | FP32/BF16/FP16、FP32 master、动态 loss scaling 等成熟路径 | FP16 params 仅 ZeRO-1/2；无 GradScaler；低精度 CE 使用 FP32 |
| kernel 依赖 | 最优路径依赖外部融合扩展 | 只使用 PyTorch reference |
| 指标日志 | 完整训练日志与跨并行组统计 | `StepOutput.metrics` 当前 rank-local；全局 scalar reduce 尚未接入 Trainer |
| 测试成本 | 大型组合矩阵，需要更多硬件/CI | 当前以 CPU/Gloo 为主；2/4/8 卡是分层验收计划 |
| 学习成本 | 高，但能看到真实生产系统 | 较低，核心通信保持可见 |

## 13. 初稿中存在的问题

这次源码对比也修正了 Nano 初稿中的四个问题：

1. 固定四维 `(DP, PP, CP, TP)` 无法把 EP 作为一等配置和 rank-order 轴。
2. 把 EP 视为 DP 内部实现细节，使用户无法直接表达 `TP×PP×CP×EP×DP` topology。
3. 单一 `dp_mesh` 不能同时服务 dense 与 expert 参数。
4. checkpoint 只承诺 DP reshard，缺少未来 topology transform 所需的逻辑 shard metadata。

修订后的独立 EP 不是简单把 `EP` 乘进 world size；关键是同时引入派生 group 和参数域，否则 topology 看似五维，梯度与 batch 语义仍然会错。

## 14. 仍需警惕的风险

### 14.1 Group 数量膨胀

五轴的所有子集有 31 种组合，不应该全部创建。`GroupPlan` 只 materialize 当前模型声明需要的 group，并对相同 ranks/backend/channel 做 alias 复用。

### 14.2 独立 EP 的双重角色

EP 对 dense 参数是 batch replica，对 expert 参数是 model shard。日志、loss scaling、data sampler、DDP 和 checkpoint 都必须使用语义化 group key，禁止随手使用 `parallel.dp`。

### 14.3 FSDP2 多参数域

在一个模型中对 dense/expert module 使用不同 mesh 是第二阶段最大技术风险。必须在正式 MoE 前实现 toy integration test；如果 FSDP2 组合限制过大，备选方案是第一版 MoE+ZeRO-3 只支持统一 dense replica mesh，随后再优化 expert domain。

### 14.4 过度抽象

GroupPlan、parameter domain、component factory 都必须保持小而具体。若一个新 GPT 功能需要先理解插件系统、服务定位器或动态 registry，项目就偏离了教学目标。

### 14.5 后端验证边界

当前已验证矩阵主要是 CPU/Gloo，全仓结果为
`482 passed, 4 skipped, 25 warnings in 462.96s`；skip 条目需要本机两张 CUDA/NCCL
GPU 或显式开启可选 profiler，warning 均为单进程 DCP 在
`torch.distributed` 未初始化时的预期提示。Ruff、lock 检查和 diff 检查也已通过。
CUDA/NCCL 已在 2×A40 和 8×A100 上完成 README 所列定向验收；完整 TinyStories 长收敛、
多节点，以及 GPU activation/optimizer/FSDP offload 的完整组合矩阵仍未系统验收。
PP 已实现 GPipe、非交错 1F1B、virtual/interleaved pipeline、P2P overlap 和
dynamic activation-shape metadata 协议；真实变长 microbatch 与长时稳定性仍未系统验收。
CP 不支持 attention dropout、packed
sequence 和 document mask。上述内容是 Nano 为降低教学和开发成本做的明确边界，
不应表述为与 Megatron 的生产路径等价。

## 15. 最终取舍

Nano-Megatron 不追求成为 Megatron-Core 的兼容实现，也不应该发明完全不同的并行数学。最终原则是：

- 并行语义向 Megatron 的成熟实现看齐。
- 依赖关系比 Megatron 当前迁移状态更严格。
- topology 比初稿更通用，EP 是独立轴。
- 性能优化只在 reference 结果稳定后加入。
- 每增加一个生产优化，都必须保留一条能读懂、能数值对照的路径。
