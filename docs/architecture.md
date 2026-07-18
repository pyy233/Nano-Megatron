# Nano-Megatron 架构设计（第一阶段：GPT）

> 状态：第一阶段架构与当前实现边界说明；仍有明确列出的 GPU 验证与 topology 转换限制。
>
> 范围：decoder-only GPT、TP（含 SP）、PP、DP（DDP/ZeRO-1/2/3）、CP、激活重计算、梯度累积、CPU offload。
>
> 暂不实现：MoE 计算与 EP token dispatch（但五维 EP topology/group 已纳入第一阶段）、量化与 FP8、多模态、Encoder/T5、弹性训练。
>
> 当前验证状态：核心正确性测试持续运行在 CPU/Gloo；2×A40 与 8×A100 上已完成 README
> 所列 CUDA/NCCL 定向验收，但完整 TinyStories 长收敛、多节点和全部组合矩阵仍未覆盖。

## 1. 设计目标

Nano-Megatron 的目标不是缩小版的生产集群平台，而是一个可以读懂、可以单步调试、又能真实组合五个并行轴的教学框架。

优先级从高到低为：

1. 并行语义正确，并能与单卡结果做数值对照。
2. 核心对象显式依赖注入，不依赖 Megatron 风格的模块级全局 `parallel_state`。
3. 每种并行方式都有独立、较小的实现边界和测试入口。
4. kernel 路径只依赖公开 PyTorch API，保持数值逻辑可读、可替换。
5. 在上述前提下再做通信重叠、融合算子等性能优化。

### 1.1 第一阶段的完成定义

- 同一 GPT 配置可以从单卡切换到 TP、TP+SP、PP、CP 和 DP。
- topology 原生包含独立 EP 轴，并能生成 CP×EP、dense/expert replica 等派生组；第一阶段不执行 MoE token dispatch。
- DP 支持 DDP、ZeRO-1、ZeRO-2、ZeRO-3；ZeRO-3 使用 PyTorch FSDP2 作为可靠底座。
- 支持非交错 1F1B pipeline 调度、梯度累积和静态形状 micro-batch。
- 支持 full/selective activation recomputation。
- 支持 optimizer state、ZeRO-3 参数/梯度以及 saved activation 的 CPU offload。
- 支持原生 PyTorch kernel 后端与 FP32/BF16/FP16 模型精度。
- 多进程 DDP/ZeRO-1/2 checkpoint 可以在相同 TP/PP/CP、不同 DP/EP replica degree
  下恢复。ZeRO-3 使用 FSDP2 canonical state + DCP，在 TP/PP/CP/EP 固定时支持改变
  DP degree，并按 `(TP, PP)` 保存独立的 dense GPT 子目录。
- EP 只预留配置和接口位置，不出现无法工作的半成品实现。

## 2. 总体原则：显式并行上下文

框架不提供以下 API：

```python
# 明确禁止
get_tensor_model_parallel_group()
get_pipeline_model_parallel_rank()
set_global_parallel_state(...)
```

唯一入口是显式构造并传递的 `ParallelContext`：

```python
parallel = ParallelContext.create(runtime, config.parallel)

model = GPTModel(
    config=config.model,
    parallel=parallel,
    kernels=kernels,
)

strategy = build_data_parallel_strategy(
    config=config.data_parallel,
    parallel=parallel,
)
```

所有会发起 collective、依赖 rank 或创建分片参数的对象，都必须在构造函数中接收 `ParallelContext` 或一个更窄的 `ParallelGroup`。普通激活函数等纯本地对象不必接收它。

PyTorch 的 default process group 仍然是 `torch.distributed` 的进程级运行时设施；本项目消除的是框架自己的全局并行状态，而不是重新实现 c10d。

## 3. 建议目录结构

```text
nano-megatron/
├── pyproject.toml
├── README.md
├── docs/
│   ├── architecture.md              # 本文：总设计与接口契约
│   ├── megatron-comparison.md        # 对 Megatron-LM 源码的架构分析
│   ├── parallelism.md               # 后续：各维度张量布局与通信图
│   └── testing.md                   # 后续：分布式测试与故障排查
├── examples/
│   ├── configs/
│   │   ├── gpt_single.yaml
│   │   ├── gpt_tp2_sp.yaml
│   │   ├── gpt_pp2.yaml
│   │   ├── gpt_cp2.yaml
│   │   └── gpt_zero3.yaml
│   └── train_gpt.py                 # 显式 API 使用示例
├── src/nano_megatron/
│   ├── __init__.py
│   ├── cli/
│   │   └── train.py                 # python -m nano_megatron.cli.train
│   ├── config/
│   │   ├── schema.py                # 所有 dataclass 配置
│   │   ├── loader.py                # YAML -> dataclass
│   │   └── validation.py            # 跨配置/拓扑校验
│   ├── distributed/
│   │   └── runtime.py               # init/destroy default process group、设备绑定
│   ├── parallel/
│   │   ├── context.py               # ParallelContext
│   │   ├── axes.py                  # ParallelAxis 与 ParallelCoordinate
│   │   ├── topology.py              # rank <-> 通用 N-D coordinate
│   │   ├── group_plan.py            # 声明式 GroupSpec/GroupPlan
│   │   ├── registry.py              # group materialization、alias 与 channel
│   │   ├── group.py                 # ParallelGroup 值对象
│   │   ├── domains.py               # DENSE/EXPERT 参数域与 replica group
│   │   ├── layout.py                # 调试用张量布局描述
│   │   ├── collectives.py           # 带 autograd 的通用 collective
│   │   └── rng.py                   # 显式 RNG stream/tracker
│   ├── tensor_parallel/
│   │   ├── layers.py                # Column/Row/Vocab parallel layers
│   │   ├── sequence_parallel.py     # SP all-gather/reduce-scatter
│   │   ├── embedding.py             # VocabParallelEmbedding
│   │   └── cross_entropy.py         # VocabParallelCrossEntropy
│   ├── context_parallel/
│   │   ├── interface.py             # ContextParallelAttention 协议
│   │   ├── all_gather.py            # 简单正确性参考实现
│   │   └── ring.py                  # P2P ring + online softmax 实现
│   ├── expert_parallel/              # 第二阶段实现，第一阶段先固定接口
│   │   ├── router.py
│   │   ├── dispatcher.py            # all-to-all reference backend
│   │   ├── experts.py
│   │   └── layer.py
│   ├── pipeline_parallel/
│   │   ├── partition.py             # layer -> stage 划分
│   │   ├── stage.py                 # PipelineStage
│   │   ├── p2p.py                   # activation/gradient send/recv
│   │   └── schedules/
│   │       ├── base.py
│   │       ├── gpipe.py             # 参考调度
│   │       └── one_f_one_b.py       # 第一阶段默认调度
│   ├── data_parallel/
│   │   ├── interface.py             # DataParallelStrategy
│   │   ├── ddp.py                   # 教学版 flat-bucket replicated DP
│   │   ├── zero1.py
│   │   ├── zero2.py
│   │   ├── zero3.py                 # FSDP2 adapter
│   │   ├── buckets.py               # flat shard / grad bucket
│   │   └── offload.py               # pinned CPU buffer 与异步拷贝
│   ├── nn/
│   │   ├── activation_checkpoint.py
│   │   ├── dropout.py               # 显式 ParallelRNG activation stream
│   │   ├── rotary.py
│   │   ├── norms.py
│   │   └── kernels/
│   │       ├── interface.py         # KernelBackend
│   │       └── torch_backend.py
│   ├── models/
│   │   └── gpt/
│   │       ├── model.py
│   │       ├── layer.py
│   │       ├── attention.py
│   │       ├── mlp.py
│   │       ├── tied_embeddings.py   # PP 首尾权重/梯度同步
│   │       └── builder.py            # 按 PP stage 构造本地模型
│   ├── training/
│   │   ├── trainer.py
│   │   ├── microbatches.py
│   │   ├── batch_router.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── checkpoint/
│   │   ├── manager.py               # PyTorch DCP adapter
│   │   ├── mapping.py               # ShardMetadata 与逻辑全局张量映射
│   │   └── manifest.py              # 配置、拓扑、版本元数据
│   └── utils/
│       ├── logging.py
│       └── memory.py
└── tests/
    ├── unit/                         # 纯 CPU；拓扑、配置、分片数学
    ├── distributed_cpu/              # torchrun + Gloo；小张量 collective
    ├── distributed_gpu/              # 每个并行维度的数值对照
    └── integration/                  # 多维组合、保存恢复、短训练
```

目录按“并行维度”拆分，而不是把所有通信集中进一个巨型文件。模型代码只表达 GPT 结构；通信发生在并行层、attention backend、schedule 和 DP strategy 中。

## 4. 可扩展五维拓扑与进程组

### 4.1 五个独立轴

TP、PP、CP、EP、DP 都是一等拓扑轴：

```text
ParallelAxis = {TP, PP, CP, EP, DP}
coordinate = (tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)
```

默认 rank order 为 `tp-cp-ep-dp-pp`，左侧变化最快；用户可以配置其他顺序，但五个 axis size 与语义不变：

```text
world_size = tp_size × pp_size × cp_size × ep_size × dp_size
```

`dp_size` 可以由 world size 和其他四轴推导，也可以显式填写并校验。第一阶段模型仍是 dense GPT，但 topology 从第一天就能创建 `EP>1` 的轴和派生组；此时 EP 对 dense GPT 只是冗余的数据副本轴，并给出提示。第二阶段加入 MoE 时不修改 rank 映射或 checkpoint topology。

这与 Megatron-LM 当前使用 dense/expert 两个 `RankGenerator` 的方式不同：Nano-Megatron 只保留一个通用五维 topology，因此允许 CP 和 EP 同时大于 1。

### 4.2 Axis group 与 replica group 必须分开

EP 是独立轴，但同一 GPU 在 dense 层和 expert 层承担的角色不同：

- 在 dense 层，EP rank 拥有相同 dense 参数、处理不同 microbatch，因此 EP 同时扩大有效 dense data parallel degree。
- 在 MoE 层，EP rank 拥有不同 local experts，并通过 all-to-all 交换 token。
- CP rank 处理同一 batch 的不同 sequence chunk，不增加 batch size，但 dense/expert 权重都要跨 CP 汇总梯度。

因此“纯 DP 轴 group”不能被当作全模型唯一梯度组。定义以下派生量：

```text
batch_replica_size   = DP × EP
dense_replica_size   = DP × EP × CP
expert_replica_size  = DP × CP
```

全局 batch 为：

```text
global_batch = micro_batch × gradient_accumulation × DP × EP
```

例如 8 个 rank 配置 `TP=1, PP=1, CP=2, EP=2, DP=2`：

| 语义 | size | 说明 |
|---|---:|---|
| EP axis group | 2 | 两个 rank 之间做 expert token dispatch |
| batch replicas | 4 | `(dp, ep)` 四种坐标，各自读取不同 microbatch |
| dense replica group | 8 | dense/router/shared-expert 权重跨 DP×EP×CP 同步 |
| expert replica group | 4 | 同一 local expert 权重跨 DP×CP 同步 |

这个例子中 CP、EP 同时大于 1，无需第二套 rank generator。

### 4.3 声明式 GroupPlan

进程组不硬编码成不断增长的 context 字段列表，而由 `GroupSpec` 声明“哪些轴变化、使用哪个 communicator channel”：

```python
class ParallelAxis(StrEnum):
    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"
    DP = "dp"


@dataclass(frozen=True)
class GroupSpec:
    key: "GroupKey"
    varying_axes: frozenset[ParallelAxis]
    select: Mapping[ParallelAxis, tuple[int, ...]] = field(default_factory=dict)
    backend: str | None = None
    channel: str = "default"


DEFAULT_GROUP_PLAN = GroupPlan(
    GroupSpec(GroupKey.TP, frozenset({ParallelAxis.TP})),
    GroupSpec(GroupKey.PP, frozenset({ParallelAxis.PP})),
    GroupSpec(GroupKey.CP, frozenset({ParallelAxis.CP})),
    GroupSpec(GroupKey.EP, frozenset({ParallelAxis.EP})),
    GroupSpec(GroupKey.DP_AXIS, frozenset({ParallelAxis.DP})),
    GroupSpec(GroupKey.TP_CP, frozenset({ParallelAxis.TP, ParallelAxis.CP})),
    GroupSpec(GroupKey.TP_EP, frozenset({ParallelAxis.TP, ParallelAxis.EP})),
    GroupSpec(
        GroupKey.DENSE_REPLICA,
        frozenset({ParallelAxis.DP, ParallelAxis.EP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.EXPERT_REPLICA,
        frozenset({ParallelAxis.DP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.BATCH_REPLICA,
        frozenset({ParallelAxis.TP, ParallelAxis.PP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.EMBEDDING,
        frozenset({ParallelAxis.PP}),
        select={ParallelAxis.PP: (0, -1)},
    ),
)
```

`channel` 允许为相同 ranks 创建独立 communicator，例如 gradient reduce 与 parameter all-gather overlap；默认 registry 会复用完全相同的 `(ranks, backend, channel)`，避免无意重复创建 NCCL group。

所有 rank 先用纯数学 `ParallelTopology` 展开同一个 `GroupPlan`，按稳定顺序 materialize 全部 group。自定义模型可以在 distributed 初始化前追加 `GroupSpec`，不必修改 `ParallelContext` 源码。

### 4.4 预定义组语义

| GroupKey | 变化轴 | 用途 |
|---|---|---|
| `TP` | TP | hidden/head/vocab 分片与 SP |
| `PP` | PP | stage activation/gradient P2P |
| `CP` | CP | attention context 交换 |
| `EP` | EP | MoE token dispatch/all-to-all |
| `DP_AXIS` | DP | 纯拓扑轴；通常不直接归约全模型梯度 |
| `TP_CP` | TP, CP | router/sequence 相关复合统计 |
| `TP_EP` | TP, EP | expert token/layout 变换 |
| `DENSE_REPLICA` | DP, EP, CP | dense/router/shared-expert 参数梯度与 ZeRO |
| `EXPERT_REPLICA` | DP, CP | local expert 参数梯度与 ZeRO |
| `BATCH_REPLICA` | TP, PP, CP | 同一 `(dp, ep)` microbatch 的模型内路由 |
| `EMBEDDING` | PP 首尾 | tied embedding/head 同步 |

未来 hierarchical CP、expert tensor parallel、partial optimizer shard 或多模块 pipeline 都通过新增 `GroupSpec` 表达，而不是加入新的全局变量。

### 4.5 参数域

```python
class ParameterDomain(StrEnum):
    DENSE = "dense"
    EXPERT = "expert"


@dataclass(frozen=True)
class ParameterPlacement:
    domain: ParameterDomain
    tensor_sharded: bool
    replica_group: GroupKey
    tensor_shard_dim: int | None = None
```

- Attention、norm、dense MLP、router、shared expert 属于 `DENSE`，replica group 为 `DENSE_REPLICA`。
- local expert weights 属于 `EXPERT`，replica group 为 `EXPERT_REPLICA`。
- 参数域由模块构造时显式注册到 `ParameterDomainRegistry`，不依赖给 `Parameter` 动态挂 `is_expert_parallel` 属性。
- `tensor_shard_dim` 让 checkpoint 能计算 TP shard 的真实 global shape/offset；Column/Vocab 参数为 dim 0，RowParallel weight 为 dim 1。

### 4.6 核心类型

```python
@dataclass(frozen=True)
class ParallelCoordinate:
    tp: int
    cp: int
    ep: int
    dp: int
    pp: int


@dataclass(frozen=True)
class ParallelGroup:
    key: GroupKey
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup
    rank: int
    size: int


class ParallelContext:
    @classmethod
    def create(
        cls,
        runtime: "DistributedRuntime",
        config: "ParallelConfig",
        group_plan: "GroupPlan | None" = None,
    ) -> "ParallelContext": ...

    @property
    def topology(self) -> "ParallelTopology": ...

    @property
    def coordinate(self) -> ParallelCoordinate: ...

    def group(self, key: GroupKey) -> ParallelGroup: ...
    def axis_group(self, axis: ParallelAxis) -> ParallelGroup: ...
    def mesh(self, key: GroupKey) -> DeviceMesh: ...
    def group_plan(self) -> GroupPlan: ...

    def is_pipeline_first_stage(self) -> bool: ...
    def is_pipeline_last_stage(self) -> bool: ...
    def pipeline_prev_rank(self) -> int | None: ...
    def pipeline_next_rank(self) -> int | None: ...
    def close(self) -> None: ...
```

为常用路径提供只读快捷属性（如 `parallel.tp`），但它们只是 `group(GroupKey.TP)` 的 typed alias。低层 TP layer 最好只接收一个 `ParallelGroup`，Trainer/模型 builder 才接收完整 `ParallelContext`，从而缩小依赖面。

`ParallelContext` 是 immutable runtime object，不提供 `use_global_groups()`、`current_context()` 或 `None` fallback。测试可以只构造纯数学 topology/group plan，不初始化 distributed。

### 4.7 RNG

`ParallelRNG` 由 context 创建并显式传入需要 dropout/初始化的模块，至少包含：

- `data`：不同 `(DP, EP)` batch replica 不同，用于 sampler/data augmentation。
- `dense_init`：同一个 dense parameter shard 在 `DENSE_REPLICA` group 内一致。
- `expert_init`：同一个 local expert shard 在 `EXPERT_REPLICA` group 内一致，不同 EP/local expert 不同。
- `activation`：按 layer、microbatch 和 global token offset 派生，使 TP/SP/CP shard 与重计算得到稳定但不重复的 dropout mask。

Trainer 使用 `global_microbatch = optimizer_step × GAS + local_microbatch`，避免每个 step 重复 dropout mask。激活重计算必须保存并恢复这些 stream 的状态，而不是只依赖默认 CUDA RNG；checkpoint 同时按完整 rank coordinate 保存 torch CPU/CUDA RNG。

## 5. 张量布局契约

统一采用 batch-first：

```text
hidden states: [micro_batch, local_sequence, hidden]
```

对全局 sequence length `S`：

```text
CP local sequence = S / cp_size
SP layer-boundary sequence = S / cp_size / tp_size
```

TP+SP 的一个 Transformer layer 数据流为：

```text
层入口：CP shard + SP shard
  -> LayerNorm（本地）
  -> TP all-gather sequence
  -> ColumnParallel QKV
  -> CP attention（query 保持 CP local，KV 跨 CP 交换）
  -> RowParallel output projection
  -> TP reduce-scatter sequence
  -> residual/dropout（本地）
  -> LayerNorm（本地）
  -> TP all-gather sequence
  -> ColumnParallel MLP up/gate
  -> RowParallel MLP down
  -> TP reduce-scatter sequence
  -> 层出口：CP shard + SP shard
```

最后一层进入 vocab-parallel head 前，先在 TP 组中退出 SP（all-gather sequence），使每个 TP rank 针对相同 token 持有不同 vocab shard；随后 vocab-parallel cross entropy 在 TP 组内归约。

建议提供只用于断言/调试的布局元数据：

```python
@dataclass(frozen=True)
class TensorLayout:
    cp_sharded: bool
    sp_sharded: bool
    tp_hidden_sharded: bool
    sequence_dim: int = 1
```

第一阶段不引入 DTensor 贯穿模型，避免教学代码同时承担自动 placement 系统的复杂度。

## 6. Tensor Parallel 与 Sequence Parallel

### 6.1 主要类

```python
class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        bias: bool = True,
        gather_output: bool = False,
        sequence_parallel: bool = False,
    ): ...

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]: ...


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        bias: bool = True,
        input_is_parallel: bool = True,
        sequence_parallel: bool = False,
    ): ...

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]: ...


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        sequence_parallel: bool,
    ): ...


class VocabParallelCrossEntropy(nn.Module):
    def __init__(self, *, parallel: ParallelContext): ...
    def forward(self, local_logits: Tensor, targets: Tensor) -> Tensor: ...
```

collective 必须是独立、可 gradcheck 的 autograd function：

```python
gather_from_sequence_parallel_region(x, group)
reduce_scatter_to_sequence_parallel_region(x, group)
copy_to_tensor_parallel_region(x, group)
reduce_from_tensor_parallel_region(x, group)
```

### 6.2 参数分片

- Attention QKV：column parallel，按 local heads 分片。
- Attention output：row parallel。
- SwiGLU gate/up：column parallel。
- MLP down：row parallel。
- Embedding/LM head：按 vocab 维分片。
- LayerNorm/RMSNorm：参数复制；激活在 SP 维切分。

第一阶段先实现 MHA；GQA 可以作为模型配置支持，但要求 `num_query_heads` 和 `num_kv_heads` 能被 TP 合法分配。若 TP 大于 KV head 数，不实现 Megatron 的 KV head replication 特例，直接在校验阶段拒绝。

## 7. Context Parallel

CP 的目标是让每个 CP rank 只长期持有 `S / cp_size` 的 query/activation，而不是简单地永久复制完整序列。

```python
class ContextParallelAttention(Protocol):
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        sequence_offset: int,
        rng: ParallelRNG,
    ) -> Tensor: ...
```

提供两个后端：

1. `all_gather`：all-gather K/V 后调用 SDPA。代码短，作为单元测试 oracle；它不是低内存实现。
2. `ring`：K/V block 在 CP 组内 P2P 轮转，通过 online softmax 合并 block 结果；这是正式训练后端。

第一阶段 CP 限制：

- 只支持 causal self-attention。
- sequence length 必须整除 CP；启用 SP 时还必须满足 `(S / CP) % TP == 0`。
- 不支持 packed sequence、document mask、滑动窗口和 cross-attention。
- `all_gather` 和 `ring` 后端都要求 attention dropout 为 0。当前统一的
  `model.dropout` 同时服务 embedding/residual/MLP 等路径，因此集中校验会在 `CP>1`
  时要求整个 `model.dropout=0`，而不是只关闭 attention 内部 dropout。

CP 只负责 attention 的上下文交换；MLP 和 LayerNorm 不在 CP 组中通信。

## 8. Pipeline Parallel

### 8.1 模型分区

```python
@dataclass(frozen=True)
class LayerPartition:
    stage: int
    start_layer: int
    end_layer: int       # exclusive
    owns_embedding: bool
    owns_final_norm: bool
    owns_lm_head: bool


class GPTModelBuilder:
    def __init__(self, components: "GPTComponentFactory"): ...

    def build_stage(
        self,
        model_config: GPTConfig,
        parallel: ParallelContext,
        kernels: KernelBackend,
    ) -> "BuiltGPTStage": ...


@dataclass(frozen=True)
class BuiltGPTStage:
    model: "GPTPipelineStage"
    partition: LayerPartition
    parameter_domains: "ParameterDomainRegistry"
```

默认按层数均匀切分，余数分给前面的 stage。第一阶段不允许空 stage；embedding 在首 stage，final norm/lm head/loss 在尾 stage。开启 tied embedding 时，首尾 stage 使用 `embedding` group 同步初始化与梯度。

### 8.2 Stage 与调度

```python
class PipelineStage(nn.Module):
    def forward(
        self,
        hidden_states: Tensor | None,
        batch: "ModelBatch",
    ) -> Tensor | "LossOutput": ...


class PipelineSchedule(Protocol):
    def forward_backward(
        self,
        *,
        stage: PipelineStage,
        microbatches: Sequence[ModelBatch],
        data_parallel: DataParallelStrategy,
        forward_only: bool = False,
    ) -> "StepOutput": ...
```

第一阶段包含：

- `GPipeSchedule`：先全 forward 再全 backward，作为易读参考和测试 oracle。
- `OneForwardOneBackwardSchedule`：warmup、steady 1F1B、cooldown，作为默认。
- `InterleavedOneForwardOneBackwardSchedule`：每个物理 PP rank 可持有多个 virtual
  model chunk，以同一 executor 执行 interleaved 1F1B。
- activation P2P 支持配置的 wire dtype、固定 shape，以及每个 microbatch 携带
  header/metadata 的 dynamic-shape 协议；`overlap_p2p=true` 使交换 request
  与 schedule 事件生命周期重叠。
- 不做 encoder-decoder 双栈调度与 Megatron-Core 级别的任意 layer layout。

P2P API 接收显式 PP group/邻居 rank，并预分配静态 shape buffer：

```python
send_forward(tensor, *, parallel, stream=None)
recv_forward(shape, dtype, *, parallel, stream=None)
send_backward(grad, *, parallel, stream=None)
recv_backward(shape, dtype, *, parallel, stream=None)
```

## 9. Data Parallel 与 ZeRO

### 9.1 统一策略接口

Trainer 不直接判断当前是 DDP 还是 ZeRO：

```python
class DataParallelStrategy(ABC):
    def configure_precision(self, precision: "PrecisionConfig") -> None: ...

    def setup(
        self,
        model: nn.Module,
        optimizer_config: "OptimizerConfig",
        parameter_domains: "ParameterDomainRegistry",
    ) -> nn.Module: ...

    @contextmanager
    def microbatch_context(self, *, is_last_microbatch: bool): ...

    def backward(self, loss: Tensor) -> None: ...
    def finalize_gradients(self) -> None: ...
    def clip_grad_norm(self, max_norm: float) -> Tensor: ...
    def optimizer_step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
```

`microbatch_context` 对 FSDP2 控制梯度同步；DDP/ZeRO-1/2 在完整 schedule 与 tied-embedding gradient 合并后，由 `finalize_gradients()` 只归约一次。strategy 按 parameter domain 分别建立 dense/expert bucket，使梯度累积和 PP schedule 都不依赖具体 DP/EP 实现。

### 9.2 各阶段语义

| 模式 | 参数 | 梯度 | Optimizer state | 第一阶段实现 |
|---|---|---|---|---|
| DDP | 复制 | 复制，整轮一次 all-reduce | 复制 | domain-aware flat bucket；不依赖 DDP reducer 隐式图状态 |
| ZeRO-1 | 复制 | 复制 | 分片 | 可读的 flat-shard AdamW |
| ZeRO-2 | 复制 | reduce-scatter 分片 | 分片 | grad bucket + sharded AdamW |
| ZeRO-3 | 分片，按需 all-gather | reduce-scatter 分片 | 分片 | PyTorch FSDP2 adapter |

DDP 也使用显式 bucket，是因为 native DDP 的 `no_sync` 必须跨越同一 forward/backward 图生命周期，难以同时适配 GPipe 逆序 drain 与 1F1B interleaving。ZeRO-1/2 自己实现，是为了清楚展示 optimizer/gradient sharding。ZeRO-3 不从零实现 module pre-forward 参数物化、backward hook、mixed precision 与 checkpoint state dict；`Zero3Strategy` 封装 FSDP2 `fully_shard`。

`precision.grad_reduce` 对 DDP/ZeRO-1/2 决定 bucket collective dtype，对 ZeRO-3 映射为 FSDP2 `MixedPrecisionPolicy.reduce_dtype`。global gradient norm 先恢复 replica-sharded subtotal，再按 TP/EP/PP 汇总，并避免重复计数 TP replicated、dense EP replica 与 PP tied embedding。

参数精度还有一个独立边界：`params=float16` 只允许 ZeRO-1/2，因为它们维护 FP32
master parameter shard；DDP 与 ZeRO-3 没有 FP32 master parameter，会在启动时拒绝该
组合。`compute=float16` 可以与 FP32 params 配合 autocast，但当前没有 GradScaler 或
动态 loss scaling，因此训练者需要自行评估 FP16 梯度下溢风险。

每个 bucket/shard plan 都带参数域：

```text
DENSE  -> parallel.mesh(GroupKey.DENSE_REPLICA)   # DP × EP × CP
EXPERT -> parallel.mesh(GroupKey.EXPERT_REPLICA)  # DP × CP
```

第一阶段只有 `DENSE` 参数，但接口和 checkpoint metadata 从一开始保存 domain。第二阶段 MoE 可以对 dense block 与 expert module 使用不同的 FSDP2 mesh，不需要重写 Trainer。

建议以单个 Transformer block 为 FSDP2 shard unit，embedding 和输出层分别作为 unit；这既控制峰值 all-gather 内存，也避免对每个 Linear 产生过细通信。

### 9.3 梯度累积

```text
global_batch_size = micro_batch_size × dp_size × ep_size × gradient_accumulation_steps
```

PP 下 `gradient_accumulation_steps` 同时就是每个 optimizer step 的 microbatch 数。配置只保留一个真值来源，`global_batch_size` 作为派生值显示，避免三个字段彼此冲突。

loss 的梯度语义按等长 CP shard 和 replica collective 得到全局平均。日志标量与梯度归约
必须区分 batch replica、CP shard 和参数域，不能把 `DP_AXIS` group 当作默认答案。

当前 `StepOutput.metrics` 只在产生 loss 的 PP 尾 stage 上，对本 rank 的 GAS
microbatch 做平均；Trainer 尚未把它接到 `reduce_scalar()`，因此不会自动跨 DP、EP 或
CP 汇总。CLI 目前只打印 step/sample/token 进度。`grad_norm` 是策略内部做过并行组
collective 的全局范数，不应与普通 rank-local metric 混为一谈。需要全局 loss/吞吐日志
时，调用方必须选择正确的语义组显式归约。

## 10. 激活重计算与 Offload

### 10.1 Activation recomputation

```python
@dataclass
class ActivationCheckpointConfig:
    mode: Literal["none", "full", "selective"] = "none"
    block_interval: int = 1
    selective_ops: tuple[str, ...] = ("attention", "mlp")
    use_reentrant: bool = False
    offload_saved_tensors: bool = False
```

- `full`：每 N 个 Transformer block checkpoint 一次。
- `selective`：只重计算 attention 或 MLP 等配置项。
- 固定使用 non-reentrant checkpoint，便于组合 FSDP2 和复杂输出。
- checkpoint 边界保存/恢复 `ParallelRNG` 状态。

### 10.2 Offload 分类

```python
@dataclass
class OffloadConfig:
    optimizer_state: bool = False
    zero3_params_and_grads: bool = False
    activations: bool = False
    pin_memory: bool = True
    non_blocking: bool = True
```

- optimizer state offload：ZeRO-1/2 的本地 Adam state 放到 pinned CPU memory；计算更新时按 bucket 搬运。
- ZeRO-3 parameter/gradient offload：两者作为一个生命周期开关，使用 FSDP2 的 CPU offload policy，不自己维护第二套参数生命周期。
- activation offload：通过 `saved_tensors_hooks`/checkpoint wrapper 把选定 saved tensor 移到 CPU。

第一阶段不实现 NVMe offload。CPU offload 默认关闭，因为它节省显存但会显著增加 PCIe 流量；配置校验会拒绝不受对应 DP mode 支持的组合。

## 11. GPT 模型与 Kernel 后端

### 11.1 GPT 主要接口

```python
class GPTModel(nn.Module):
    def __init__(
        self,
        config: "GPTConfig",
        *,
        parallel: ParallelContext,
        kernels: "KernelBackend",
        rng: "ParallelRNG | None" = None,
    ): ...

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> "GPTOutput": ...


class GPTLayer(nn.Module):
    def __init__(
        self,
        config: "GPTConfig",
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: "KernelBackend",
        cp_attention: "ContextParallelAttention",
        rng: "ParallelRNG",
    ): ...
```

模型默认：RoPE、RMSNorm、SwiGLU、causal self-attention、tied embedding 可配置。第一阶段不实现 learned absolute position embedding、ALiBi、cross-attention 和 MoE。

### 11.2 KernelBackend

```python
class KernelBackend(Protocol):
    name: str

    def linear(self, in_features: int, out_features: int, **kwargs) -> nn.Module: ...
    def rms_norm(self, hidden_size: int, eps: float) -> nn.Module: ...
    def local_attention(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor: ...
```

`TorchKernelBackend` 使用 `nn.Linear`、项目 RMSNorm 和 PyTorch SDPA；它是当前唯一实现，
也是所有并行组合的数值参考。

第一阶段 precision 只允许 FP32、BF16、FP16。配置 schema 不提供 FP8/INT8/INT4 字段；它们归入后续量化阶段。
此外，`compute=float32` 要求 `params=float32`；低精度参数配 FP32 compute 会在集中
校验阶段拒绝，而不是隐式提升参数精度。
`params=float16` 只支持 ZeRO-1/2；DDP/ZeRO-3 会拒绝。FP16 compute 路径没有
GradScaler/动态 loss scaling。Vocab-parallel cross entropy 对 FP16/BF16 logits 在
FP32 中完成 max、exp、sum 和 loss，再把 logits gradient 转回输入 dtype。

### 11.3 组件扩展接口

参考 Megatron-Core `ModuleSpec` 的可替换能力，但不采用任意动态 import + kwargs 字典。使用小型 typed factory：

```python
class GPTComponentFactory(Protocol):
    def build_attention(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
    ) -> nn.Module: ...

    def build_mlp(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        parameter_domains: ParameterDomainRegistry,
    ) -> nn.Module: ...

    def build_norm(self, config: GPTConfig, *, kernels: KernelBackend) -> nn.Module: ...
```

第一阶段提供 `DenseGPTComponents`；第二阶段增加 `MoEGPTComponents`，只替换目标 layer 的 MLP builder。特殊 attention、不同 norm 或新 kernel backend 可以组合替换，不需要修改 `GPTLayer`，同时保留静态类型和清楚的构造调用链。

## 12. 配置设计

配置使用标准库 dataclass 定义，以 YAML 加载；不引入巨型命令行参数表。CLI 只提供配置文件路径和少量 `key=value` override。

### 12.1 顶层配置

```python
@dataclass
class TrainConfig:
    distributed: DistributedConfig
    parallel: ParallelConfig
    model: GPTConfig
    precision: PrecisionConfig
    kernels: KernelConfig
    data_parallel: DataParallelConfig
    pipeline: PipelineConfig
    activation_checkpoint: ActivationCheckpointConfig
    offload: OffloadConfig
    optimizer: OptimizerConfig
    lr_scheduler: LearningRateSchedulerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    validation: ValidationConfig
    wandb: WandbConfig
```

### 12.2 可选配置总表

| 配置块 | 主要字段 | 默认/说明 |
|---|---|---|
| `distributed` | `backend`, `timeout_minutes`, `device` | 默认均为 `auto`：有 CUDA 时选择 NCCL/CUDA，否则选择 Gloo/CPU |
| `parallel` | `tensor`, `pipeline`, `context`, `expert`, `data`, `order`, `sequence_parallel` | 五个独立轴；DP 可推导；默认 order 为 `tp-cp-ep-dp-pp` |
| `model` | `layers`, `hidden_size`, `ffn_hidden_size`, `heads`, `kv_heads`, `seq_length`, `vocab_size`, `rope_theta`, `dropout`, `tie_embeddings` | decoder-only GPT |
| `precision` | `params`, `compute`, `grad_reduce` | 推荐 BF16 compute、FP32 reduce；FP16 params 仅 ZeRO-1/2；FP16 compute 无动态 loss scaling |
| `kernels` | `backend` | 当前固定为 `torch`；无 FP8 |
| `data_parallel` | `mode`, `bucket_bytes`, `overlap_grad_reduce`, `reshard_after_forward` | `ddp/zero1/zero2/zero3`；DDP/ZeRO-1/2 支持 bucket gradient overlap，ZeRO-3 由 FSDP2 管理通信 |
| `pipeline` | `schedule`, `activation_dtype`, `overlap_p2p`, `virtual_stages_per_rank`, `dynamic_activation_shapes` | `gpipe/1f1b/interleaved_1f1b`；支持 P2P overlap 和动态 shape metadata 协议 |
| `activation_checkpoint` | `mode`, `block_interval`, `selective_ops` | 默认关闭 |
| `offload` | `optimizer_state`, `zero3_params_and_grads`, `activations`, `pin_memory` | 默认全关 |
| `optimizer` | `name`, `lr`, `betas`, `eps`, `weight_decay`, `clip_grad_norm` | 第一阶段只做 AdamW |
| `lr_scheduler` | `schedule`, `warmup_steps`, `decay_steps`, `min_lr` | constant/cosine；按已完成 optimizer step 恢复 |
| `training` | `micro_batch_size`, `gradient_accumulation_steps`, `max_steps`, `seed`, `log_interval` | global batch 派生 |
| `checkpoint` | `directory`, `save_interval`, `async_save`, `keep_last` | DDP/ZeRO-1/2 多进程使用 rank-local shard；ZeRO-3 使用 canonical DCP 并支持 DP resize；支持只在完成后发布的 async save |
| `data` | `path`, `mmap_path`, `text_path`, `tokenizer`, `num_workers`, `shuffle` | fixed-length token blocks；multi-epoch 精确恢复 |
| `validation` | `interval`, `batches`, `data` | 固定 batch 数的周期性 token-weighted perplexity |
| `wandb` | `enabled`, `project`, `entity`, `name`, `group`, `tags`, `mode`, `run_id` | 可选 tracking extra；仅主 rank |

### 12.3 示例 YAML

```yaml
distributed:
  backend: nccl
  timeout_minutes: 30

parallel:
  tensor: 2
  pipeline: 2
  context: 1
  expert: 1
  data: null              # 由 WORLD_SIZE / (TP×PP×CP×EP) 推导
  order: [tp, cp, ep, dp, pp]
  sequence_parallel: true

model:
  layers: 12
  hidden_size: 768
  ffn_hidden_size: 2048
  heads: 12
  kv_heads: 12
  seq_length: 2048
  vocab_size: 50304
  rope_theta: 10000.0
  dropout: 0.0
  tie_embeddings: true

precision:
  params: bfloat16
  compute: bfloat16
  grad_reduce: float32

kernels:
  backend: torch

data_parallel:
  mode: zero3
  bucket_bytes: 268435456
  overlap_grad_reduce: false
  reshard_after_forward: true

pipeline:
  schedule: 1f1b
  overlap_p2p: false

activation_checkpoint:
  mode: full
  block_interval: 1

offload:
  optimizer_state: false
  zero3_params_and_grads: false
  activations: false

optimizer:
  name: adamw
  lr: 0.0003
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  clip_grad_norm: 1.0

lr_scheduler:
  schedule: cosine
  warmup_steps: 500
  decay_steps: 10000
  min_lr: 0.00003

training:
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 10000
  seed: 1234
  log_interval: 10

checkpoint:
  directory: checkpoints/gpt
  save_interval: 500
  async_save: false
  keep_last: 2

validation:
  interval: 100
  batches: 20
  data:
    mmap_path: data/validation.mmap
    shuffle: false

wandb:
  enabled: false
  project: nano-megatron
  mode: online
```

### 12.4 必须在启动时完成的校验

- `world_size == TP × PP × CP × EP × DP`，且 `order` 恰好包含五个轴一次。
- `precision.compute=float32` 时 `precision.params` 也必须为 `float32`。
- `precision.params=float16` 只允许 `data_parallel.mode=zero1/zero2`；DDP/ZeRO-3
  因没有 FP32 master parameter 而拒绝。
- `hidden_size % TP == 0`、`ffn_hidden_size % TP == 0`。
- `heads % TP == 0`、`kv_heads % TP == 0`。
- `seq_length % CP == 0`；SP 开启时 `(seq_length / CP) % TP == 0`。
- `vocab_size` 自动 pad 到 TP 的倍数，manifest 中同时保存原始/填充大小。
- `layers >= PP`，且 layer partition 不为空。
- 1F1B 建议 `gradient_accumulation_steps >= PP`；小于 PP 时给出低利用率警告。
- ZeRO-3 对 dense 参数使用 `DP×EP×CP` replica mesh；mesh size=1 时允许退化但发出提示。
- optimizer offload 仅支持 ZeRO；ZeRO-3 offload 走 FSDP2 policy。
- CP 第一阶段拒绝 packed/document mask。
- CP>1 时 `model.dropout` 必须为 0，因为正式 ring 后端尚不实现 attention dropout；CP=1 自动使用 local attention，不经过 CP wrapper。
- `overlap_grad_reduce`、`overlap_p2p`、`async_save` 在第一阶段若开启会集中报错，不会静默忽略。
- dense GPT 允许 `expert > 1` 以验证 topology，但提示它等价于额外 dense batch replicas；只有配置 MoE layer 时才启用 token dispatch/local experts。
- FP8/INT8/INT4 等未知量化字段直接按 schema 未支持字段报错。

## 13. 最终用户如何使用

### 13.1 高层 CLI

```bash
uv run python -m nano_megatron.cli.train \
  --config examples/configs/gpt_single_cpu.yaml \
  --max-steps 1
```

CLI 内部只做配置加载和对象装配，不隐藏并行对象为全局变量。

### 13.2 显式 Python API

```python
from nano_megatron.checkpoint import CheckpointManager
from nano_megatron.config import load_config
from nano_megatron.data import build_train_dataloader
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.models.gpt import DenseGPTComponents, GPTModelBuilder
from nano_megatron.nn.kernels import build_kernel_backend
from nano_megatron.parallel import ParallelContext, ParallelRNG
from nano_megatron.training import Trainer


config = load_config("examples/configs/gpt_single_cpu.yaml")

with DistributedRuntime(config.distributed) as runtime:
    parallel = ParallelContext.create(runtime, config.parallel)
    rng = ParallelRNG.from_context(config.training.seed, parallel)
    kernels = build_kernel_backend(config.kernels, parallel=parallel)

    built = GPTModelBuilder(components=DenseGPTComponents()).build_stage(
        model_config=config.model,
        parallel=parallel,
        kernels=kernels,
        rng=rng,
    )

    data_parallel = build_data_parallel_strategy(
        config=config.data_parallel,
        offload=config.offload,
        parallel=parallel,
        parameter_domains=built.parameter_domains,
    )

    checkpoint = CheckpointManager(
        config=config.checkpoint,
        parallel=parallel,
    )
    data = build_train_dataloader(config, parallel)

    trainer = Trainer(
        config=config,
        model=built.model,
        parallel=parallel,
        data_parallel=data_parallel,
        checkpoint=checkpoint,
        data_iterator=iter(data),
        rng=rng,
    )
    trainer.fit()

    parallel.close()
```

### 13.3 Batch 路由

每个 `(dp_rank, ep_rank)` 对应一个独立 batch replica。其 data leader 读取样本，再在 `BATCH_REPLICA` group（变化 TP×PP×CP，固定 DP×EP）内路由必要的 token/label metadata。第一阶段优先采用简单、可验证的 broadcast；数据 batch 相比模型 activation 很小。随后：

- 首 PP stage 消费 input ids。
- 尾 PP stage 消费 labels。
- CP 按 sequence 切 token/position/label。
- TP 中间层通过 SP 管理 activation；用户数据加载器无需理解 SP。

## 14. Checkpoint

`CheckpointManager` 按 DP strategy 选择布局与物理存储：DDP/ZeRO-1/2 使用
`ShardMetadata` 描述 TP/PP local shard，ZeRO-3 使用 FSDP2 canonical state；并不是
所有分布式模式都走同一个 DCP 路径：

```python
class CheckpointManager:
    def save(
        self,
        step: int,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        trainer_state: Mapping[str, Any],
        rng: ParallelRNG | None = None,
    ) -> Path: ...

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        rng: ParallelRNG | None = None,
    ) -> "TrainerState": ...
```

DDP/ZeRO-1/2 路径增加一层轻量逻辑映射，避免 checkpoint API 被当前 topology 锁死：

```python
@dataclass(frozen=True)
class ShardMetadata:
    logical_key: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    sharded_axes: tuple[ParallelAxis, ...]
    parameter_domain: ParameterDomain
    replica_coordinate: tuple[int, ...]
```

DDP/ZeRO-1/2 的 model tensor 先产生 `ShardedState`。单进程保存优先使用 DCP，
不可用时回退 `torch.save`；多进程使用共享文件系统上的 rank-local `torch.save`
payload，每个 `(TP, PP, EP)` 逻辑 shard 只选择一个 `(CP=0, DP=0)` writer，避免
global rank 0 覆盖 PP/TP/ZeRO 各 rank 的不同状态。

ZeRO-3 使用另一条物理路径：通过 PyTorch distributed checkpoint state-dict API 导出
FSDP2 canonical model state 和 canonical optimizer state，再由 DCP 保存/加载。dense
GPT 为每个 `(TP, PP)` coordinate 使用独立目录：

```text
fsdp2_states/
├── dense_tp0000_pp0000/
├── dense_tp0001_pp0000/
└── dense_tp0000_pp0001/
```

同一目录内由对应 FSDP2 replica group 协作保存。第一阶段 ZeRO-3 restore 要求
TP/PP/CP/EP degree 不变，但 DCP canonical state 支持 DP degree resize；已经覆盖
DP2→DP1→DP2 的恢复。未来 expert domain 的目录 key 还会包含 EP coordinate。

所有模式都会为每个完整 coordinate 另存 torch/`ParallelRNG` runtime state。

manifest 保存：

- schema/framework/PyTorch 版本。
- 完整模型与训练配置。
- TP/PP/CP/EP/DP topology、rank order、GroupPlan 和 layer partition。
- DDP/ZeRO-1/2 每个参数的 domain、逻辑 global shape、shard axis/offset 与 replica
  coordinate；ZeRO-3 的 canonical tensor/optimizer metadata 由 DCP state-dict API 管理。
- vocab padding、dtype、global step、consumed samples/tokens。
- 每个完整 `(tp, pp, cp, ep, dp)` coordinate 的 torch CPU/CUDA RNG 与 `ParallelRNG` state，以及 consumed sample/token 位置。

对于多进程 DDP/ZeRO-1/2，第一阶段 dense GPT 保证相同 TP/PP/CP 下改变 DP/EP
replica degree 的恢复；新增 replica 若没有旧 RNG coordinate，会保留当前初始化状态并
给出 warning，不能承诺 bitwise continuation。改变 TP、PP 或 CP 涉及参数/layout
变换，第一阶段明确拒绝。MoE 启用后改变 EP degree 需要 expert remap planner，属于
第二阶段。ZeRO-3 的 manifest 固定 TP/PP/CP/EP，只把 DP 标记为 resizable；新增 DP
coordinate 的 RNG 同样采用“保留当前初始化状态并 warning”的策略。

## 15. 相对 Megatron-Core 的主要简化

源码对比基于本机 Megatron-LM `main@779c5b748`，详细证据和优劣分析见 [megatron-comparison.md](megatron-comparison.md)。

本设计吸收了 Megatron 的正交 rank generator、显式 `ProcessGroupCollection`、ModuleSpec 可替换组件、dense/expert 参数分组和 sharded checkpoint metadata；同时去掉全局 fallback、双 topology 和大而全配置。

| 方面 | Nano-Megatron 第一阶段 | Megatron-Core 类生产实现 |
|---|---|---|
| 模型 | 单一 decoder-only GPT | GPT、T5、Mamba、多模态、MoE 等 |
| 并行状态 | immutable `ParallelContext`，构造时必须显式传入，无 fallback | 正从全局 `parallel_state` 迁移到 `ProcessGroupCollection`，两套路径并存 |
| rank layout | 单一通用 TP/PP/CP/EP/DP 五轴 topology | 通用 `RankGenerator`，但 dense/expert 使用两个 generator，单个 generator 禁止 CP、EP 同时大于 1 |
| 进程组扩展 | 声明式 `GroupPlan`，按 axis 组合和 channel 生成 | `ProcessGroupCollection` 枚举大量字段，兼容成熟但字段持续增长 |
| 参数域 | 显式 `ParameterDomainRegistry` | 通过 parameter attribute/buffer key 区分 expert 参数 |
| 组件替换 | 小型 typed `GPTComponentFactory` | 通用 `ModuleSpec`/submodule spec，能力更强但动态性更高 |
| TP 层 | Attention/MLP/embedding/head 的核心切法 | 更多特殊层、融合和兼容分支 |
| SP | 只支持与 TP 配套的标准路径 | 更多模型/算子组合 |
| PP | GPipe + 非交错/interleaved 1F1B；virtual stage、P2P overlap、dynamic-shape 协议 | 更灵活的自定义 layout、encoder-decoder 调度等 |
| CP | causal self-attention；reference + ring | 多种通信算法、mask/packed sequence 优化 |
| EP | 第一阶段只建 topology/group/domain 接口；第二阶段 reference all-to-all MoE | 多 dispatcher、shared expert、overlap、DeepEP/NCCL EP 等完整实现 |
| DP | 教学版 bucket DDP/ZeRO-1/2；按参数域使用 FSDP2 ZeRO-3 | 约 3000 行 distributed optimizer 与深度 overlap/reshard 优化 |
| Kernel | PyTorch reference | 外部扩展与大量融合 kernel |
| Precision | FP32/BF16/FP16；FP16 params 仅 ZeRO-1/2；无 GradScaler；低精度 CE 用 FP32 | FP8、动态 loss scaling、更多 mixed precision/量化路径 |
| Offload | CPU optimizer/param/grad/activation | 更复杂异步流水与存储层方案 |
| Checkpoint | DDP/ZeRO-1/2 使用轻量 shard metadata/rank-local payload；ZeRO-3 使用 canonical DCP，固定 TP/PP/CP/EP、支持 DP resize | 成熟 `ShardedTensor`/strategy/resharding 体系 |
| 数据 | fixed-length token block | 大规模 blend、packed、复杂采样与恢复 |
| 通信优化 | 默认同步、逐项增加 overlap | 大量 overlap、bucket 调优、拓扑优化 |
| 运行环境 | 单机优先 | 大规模多机、容错和平台集成 |

具体保留的教学价值：

- TP、SP、CP、EP、PP、ZeRO 各自的 axis、参数域和 collective 在代码中可见。
- topology、group plan 和 process-group materialization 是三个独立层，可以只测 rank 数学。
- ZeRO-1/2 保留手写的 shard/bucket/update 路径，便于比较三阶段内存变化。
- ZeRO-3 借 FSDP2 保证生命周期正确，但 adapter 会把 all-gather/reduce-scatter 时机记录到 trace，便于学习。
- reference backend 永远存在；优化实现必须与 reference 做 loss/gradient 对照。
- 不使用任意动态 import registry；普通构造函数、enum key 和小型 protocol 足够。

明确牺牲：第一阶段吞吐不会追平 Megatron-Core，尤其是默认不开通信重叠、没有大量 fused kernel、没有面向大集群的通信调优。项目成功标准是“能解释且能验证”，不是 benchmark 排名。

## 16. 测试策略

当前全量 CPU/Gloo 验收结果为 `482 passed, 4 skipped, 25 warnings in 462.96s`。
4 项 skip 是本机缺少两张 CUDA/NCCL GPU 或未开启可选 profiler 用例；25 条 warning
均为单进程 DCP 在 `torch.distributed` 未初始化时的预期提示。Ruff、lock 检查和
`git diff --check` 也已通过。两卡和八卡条目同时记录分层验收计划；已完成的真实硬件项目会显式注明，
未注明的仍不能视为已验证。

### 16.1 不需要 GPU 的测试

- 配置解析、校验和错误信息。
- 任意 rank order 下五轴 rank/coordinate 双向映射、GroupPlan 展开和所有 group 成员列表。
- `DENSE_REPLICA`/`EXPERT_REPLICA`/`BATCH_REPLICA` 的参数域与 batch 数学。
- layer partition、microbatch schedule 事件序列。
- 参数 shard、bucket padding 和 checkpoint manifest。
- 使用 Gloo + 多进程的小张量 autograd collective 测试。

### 16.2 两卡数值对照

每项都用极小 GPT、固定 seed，与单卡 reference 比较 loss、参数梯度和一步 optimizer 后参数：

- TP=2，SP off/on。
- PP=2，GPipe 与 1F1B。
- CP=2，ring 对 all-gather reference。
- EP=2 的 axis/group/all-to-all 小张量测试；第一阶段 dense GPT 下验证它增加 batch replica。
- DP=2，DDP/ZeRO-1/2/3。
- 梯度累积等价于大 batch。
- activation recomputation on/off。
- 各种 CPU offload on/off。
- DDP/ZeRO-1/2 保存、销毁进程、恢复并继续一步。
- ZeRO-3 canonical DCP round-trip、DP2→DP1→DP2 reshard，以及 PP stage 独立子目录。

### 16.3 四卡组合测试

- TP2 × DP2（含 SP + ZeRO-3）。
- PP2 × DP2。
- CP2 × DP2。
- TP2 × PP2。
- TP2 × CP2。
- EP2 × DP2（dense/expert replica group 不混淆）。
- EP2 × CP2（CP、EP 同时大于 1）。
- TP2 × EP2。

重点检查不同 group 不串线、loss scaling 正确，以及 checkpoint 沿各模式声明支持的
replica 轴重分片；ZeRO-3 当前只允许改变 DP degree。

### 16.4 八卡里程碑测试

2026-07-18 已在单节点 8×A100-SXM4-80GB、Torch 2.6.0+cu124、NCCL 2.21.5 上完成
以下五组 30-step BF16 短收敛、validation 和 checkpoint/load：

- TP2 × PP2 × CP2，DP1。
- TP2 × PP2 × DP2，CP1。
- TP2 × CP2 × DP2，PP1。
- TP2 × CP2 × EP2。
- CP2 × EP2 × DP2。

所有组合固定 global batch=8；最复杂的 TP2×PP2×CP2 还从 step 15 恢复并精确复现到
step 30。该结果是分布式正确性与短收敛里程碑，不等同于完整 TinyStories 长训练验收；
EP 组合仍是 dense batch-replica 语义，不包含 MoE token dispatch。

同日在同一硬件/软件环境中完成了下列扩展验收，全程未设
`NCCL_P2P_DISABLE`：

- `TP2×PP2×CP2` 使用默认 NCCL P2P transport 运行 10 步，未出现旧
  A40 Pod 上的连续 collective 卡死。
- `TP2×CP2×DP2` 使用 `context_parallel.backend=ring` 运行 10 步；
  先前五组 30-step 基线使用的是 all-gather CP。
- `TP2×PP2×DP2` 分别验证 1F1B P2P overlap、VP2/interleaved 1F1B，
  以及 dynamic activation-shape header/metadata/接收端分配协议。dynamic case
  的实际数据固定为 seq=64，没有生成跨 microbatch 变长序列。

ZeRO 矩阵每格使用 global batch=8，运行 10 步训练和 step 5/10 validation。
下表是 train loss 的 step 1→10：

| 模式 | TP2×PP2×CP2 | TP2×PP2×DP2 | TP2×CP2×DP2 | TP2×CP2×EP2 | CP2×EP2×DP2 |
|---|---:|---:|---:|---:|---:|
| ZeRO-1 | 4.2149→2.3288 | 4.2149→2.3301 | 4.1916→2.4351 | 4.1916→2.4351 | 4.1957→2.2958 |
| ZeRO-2 | 4.2149→2.3288 | 4.2149→2.3301 | 4.1916→2.4351 | 4.1916→2.4351 | 4.1957→2.2958 |
| ZeRO-3 | 4.2149→2.3450 | 4.2149→2.3488 | 4.1916→2.4535 | 4.1916→2.4535 | 4.1957→2.3145 |

15 格全部完成且 validation loss 下降。代表拓扑 `TP2×PP2×DP2` 还分别完成
ZeRO-1/2/3 step 5 checkpoint 保存、新进程加载和 step 6 续训；ZeRO-3
async save 另外验证了 `.complete` 发布。过程中暴露出 Torch 2.6 DCP 将
subgroup coordinator 同时当作 group rank 和 global rank 的兼容性问题；当 DP
group 不含 global rank 0 时保存失败。checkpoint manager 现在仅对该旧实现
回填新版 PyTorch 已有的 global-coordinator 映射，修复后的真实 8 卡恢复已通过。

若要让 TP、PP、CP、EP、DP 五个维度同时都大于 1，最小是 32 rank（2×2×2×2×2）。第一阶段不要求这个昂贵组合；纯 CPU 五维 topology 穷举、GPU 三维组合和所有关键两两组合足以发现绝大多数 group/layout 错误。

## 17. 最低成本 GPU 建议

### 17.1 结论与当前验证状态

编写架构、数值逻辑和 Gloo 多进程测试不需要 GPU。CUDA/NCCL 正确性验证的最低可行方案是按需租同一台机器上的 **2 张 Ampere 或更新架构、每张 12 GB 以上的 NVIDIA GPU**；更稳妥、性价比更好的建议是 **2×A10/RTX 3090 24 GB**。8 GB 卡也能跑极小 correctness case，但给 FSDP2 与 offload 留出的调试余量较小。

截至当前版本，除 CPU/Gloo 外，已完成 2×A40 上的主要单维/ZeRO/offload 定向验证，
以及 8×A100 上五组多维拓扑的短收敛、扩展 pipeline/CP/NCCL 路径和
ZeRO-1/2/3 × 五拓扑的笛卡尔积。尚未完成的是完整 TinyStories 长收敛、
多节点、这五组之外的更大拓扑穷举，以及系统性长时间性能/稳定性验收。
下面保留的是一般租卡建议；具体已完成矩阵以 16.4 和 README 为准。

不建议把 T4/V100 作为主开发卡：它们可以运行部分 FP16 路径，但不适合作为统一的
BF16 验证环境。A100/H100 没有必要，除非后续做性能优化。

### 17.2 分阶段租用

| 阶段 | GPU | 建议时长/用途 |
|---|---|---|
| 架构、配置、拓扑、CPU collective | 0 | 本地 CPU 完成 |
| GPT、重计算、累积、单卡 offload | 1×12/24 GB | 单卡功能与内存测试 |
| TP/SP、PP、CP、ZeRO 分别验收 | 2×12/24 GB Ampere | 日常主要多卡环境；推荐 A10/3090 24 GB |
| 两维组合回归 | 4×A10 24 GB | 每个里程碑短租 |
| 三维组合发布测试（含 EP axis） | 8×A10 24 GB | 可选发布里程碑；预算紧时不阻塞第一阶段 |

如果预算很紧，可以长期只保留 CPU 环境，按功能完成度分几次租 2 卡；4 卡只在里程碑集中跑，8 卡属于发布前可选项。无需为五轴同时非 1 而租 32 卡。

### 17.3 主机配置

对 2×A10 24 GB：

- 16 vCPU 以上。
- 64 GB RAM 起步；大量 parameter/optimizer/activation offload 时推荐 128 GB。
- 200 GB 以上本地 NVMe，存环境、数据和多个 sharded checkpoint。
- 两张 GPU 必须在同一节点；PCIe 足够做正确性开发，NVLink 不是硬要求。
- CUDA/NCCL 与 PyTorch 版本匹配。

测试模型应控制在 50M～300M 参数、sequence 256～2048；验证并行正确性看的是切分和通信，不需要用 7B 模型烧钱。

## 18. 建议实现顺序

1. 五轴配置、`DistributedRuntime`、`ParallelTopology`、`GroupPlan`、`ParallelContext`。
2. 单卡 GPT + PyTorch kernel backend，建立数值 reference。
3. TP 基础层、vocab loss，再加入 SP。
4. `ParameterDomainRegistry`、DDP、梯度累积、ZeRO-1/2。
5. FSDP2 ZeRO-3 与 DCP checkpoint。
6. activation recomputation 和三类 CPU offload。
7. PP 的 GPipe reference 与 1F1B。
8. CP all-gather reference 与 ring backend。
9. 四卡/八卡组合测试与文档收尾。

第二阶段再实现 MoE router、local experts 和 EP all-to-all dispatcher；由于 topology、group plan、参数域与 checkpoint metadata 已就位，不需要修改第一阶段 Trainer 的核心接口。

这个顺序始终保留“上一阶段可作为下一阶段 oracle”的路径，出现数值差异时可以定位到单个并行维度，而不是同时调试整个五维系统。
