# nano-megatron 开发过程中的 Bug 修复复盘

## 1. 这份复盘的依据

这份文档不是根据代码命名猜测问题，而是交叉检查了三类记录：

1. 项目根目录下 planning-with-files 生成的 `task_plan.md`、`progress.md` 和
   `findings.md`；
2. Git 提交历史，重点是 `3b6a125`、`2d9f93e` 和 `5f372f7`；
3. 修复提交新增的回归测试，以及当前源码中的兼容性注释。

现有 planning 文件主要记录最近一次源码架构梳理，并不是最初开发阶段的完整
流水账。因此，本文只把有明确提交差异或回归测试支持的问题列为“已解决 Bug”。
按这个标准，可以确认 8 个项目问题：1 个文档发布问题、7 个运行时或兼容性问题。

## 2. 总览

| 编号 | 范围 | 问题 | 主要后果 | 修复提交 |
|---|---|---|---|---|
| 1 | README | 链接指向未发布的文档 | 仓库页面出现失效导航 | `3b6a125` |
| 2 | 全局梯度范数 | 标量 collective 没有按通信后端选择设备 | NCCL/Gloo 下设备不匹配，梯度裁剪失败 | `2d9f93e` |
| 3 | PP/FSDP | 流水线 stage 可以跨分片边界返回 Python 包装对象 | FSDP/调度器边界契约不稳定 | `2d9f93e` |
| 4 | PP P2P | 单向发送和接收使用阻塞式 API | GPU/NCCL 路径与组合交换语义不一致，存在卡死风险 | `2d9f93e` |
| 5 | NCCL subgroup | 创建子组时没有绑定 `device_id` | 重叠 communicator 首次使用顺序不一致，可能卡死 | `5f372f7` |
| 6 | NCCL warmup | 只预热 P2P 通道，没有先预热模型通信组 | 模型 collective 与 P2P communicator 初始化互相等待 | `5f372f7` |
| 7 | FSDP2 | 只从 PyTorch 2.6 的公开命名空间导入 | 2.4/2.5 中实际存在的 FSDP2 API 被误判为不可用 | `5f372f7` |
| 8 | DCP | 无条件向 checkpoint API 传递 `no_dist` | 旧版函数签名直接抛出 `TypeError` | `5f372f7` |

## 3. 各问题的发现与修复

### 3.1 README 链接指向未发布文档

初始提交的 README 链接到了 `docs/architecture.md` 和
`docs/megatron-comparison.md`，但这两个文件没有随提交发布。对仓库使用者来说，
这就是两个失效链接。

修复方式很直接：提交 `3b6a125` 删除了尚未发布的链接，没有让 README 对工作区
之外的文件作出承诺。后续 planning 记录也确认，`docs/` 当时只是本地未跟踪目录，
所以这个修复本质上是把“本地存在”与“仓库已经发布”区分开。

这里的经验是：README 只能链接版本库中实际可获得的内容；本地草稿不能被当成
已发布文档。

### 3.2 全局梯度范数的 collective 设备选择错误

梯度裁剪需要在 TP、EP、PP 等进程组之间归并平方范数。旧实现直接在当前标量所在
设备上执行 `all_reduce`，隐含假设是“标量设备一定适合当前进程组后端”。这个假设
在混合后端或分片场景中不成立：

- NCCL collective 要求 CUDA Tensor；
- Gloo 标量归并应使用 CPU Tensor；
- 标量为了通信被移动到新设备后，原调用方不能再依赖原地修改原 Tensor。

修复在 [`_common.py`](../src/nano_megatron/data_parallel/_common.py#L107) 中增加
了后端感知的设备选择：NCCL 使用当前 runtime 的 CUDA device，Gloo 使用 CPU，
其他后端保留原设备。归并函数也改为返回实际参与通信的 Tensor，调用方接住返回值，
最后再把范数移回原结果设备。

对应测试
[`test_norm_scalar_collective_device_follows_the_explicit_backend`](../tests/unit/test_data_parallel.py#L60)
固定了 NCCL/Gloo 的设备选择规则。

### 3.3 流水线 stage 返回结构化对象，破坏分片边界契约

旧接口允许最后一个 pipeline stage 返回 `LossOutput`，其中再包装 loss Tensor 和
metrics。普通 Python 调用能处理这种对象，但 FSDP/sharding 边界需要一个简单、明确
的 Tensor 数据流。允许任意 Python 包装对象会让调度器、autograd 和分片模块对
“真正参与反向传播的 Tensor 在哪里”产生不同理解。

修复把 [`PipelineStage`](../src/nano_megatron/pipeline_parallel/stage.py) 的输出统一为
Tensor，并在
[`extract_loss`](../src/nano_megatron/pipeline_parallel/schedules/base.py#L102) 中强制
最后一个 stage 返回零维 loss Tensor。loss metric 由调度器从这个标量派生，不再让
stage 穿过分片边界携带自定义包装对象。

回归测试分别验证了两条约束：

- 结构化输出会被拒绝：
  [`test_gpipe_rejects_structured_outputs_at_the_sharding_boundary`](../tests/unit/test_pipeline_parallel.py#L303)；
- 非标量 Tensor 不能充当最后 stage 的 loss：
  [`test_gpipe_requires_a_scalar_loss_from_the_last_stage`](../tests/unit/test_pipeline_parallel.py#L338)。

这个修复牺牲了 stage 自定义 metrics 的便利，但换来了清晰的分片和 autograd 边界。

### 3.4 单向 P2P 操作仍在走阻塞式 send/recv

最初的单向 `send_forward`、`recv_forward`、`send_backward` 和 `recv_backward`
直接调用 `dist.send`/`dist.recv`，而组合式流水线交换已经使用 `P2POp` 和
`batch_isend_irecv`。同一个 communicator 因此存在两套生命周期和等待语义，在
NCCL GPU 流水线中容易出现操作配对或初始化顺序问题。

`2d9f93e` 先把单向操作统一到 batched P2P API；后续 overlap 实现进一步把所有入口
收敛到当前的
[`start_exchange`](../src/nano_megatron/pipeline_parallel/p2p.py#L464) 生命周期，
同步接口只负责等待对应 request。底层发送现在统一经过
[`batch_isend_irecv`](../src/nano_megatron/pipeline_parallel/p2p.py#L439)。

测试
[`test_individual_p2p_operations_use_the_batched_api`](../tests/unit/test_pipeline_p2p.py#L59)
显式禁止回退到阻塞式 `dist.send`/`dist.recv`，并检查每个异步 request 都被等待。

### 3.5 NCCL 子组没有在创建时绑定本地 CUDA device

PyTorch 2.6 可以在 `new_group` 中接收 `device_id`，从而立即形成 NCCL
communicator。旧实现没有传这个参数，TP/PP 等重叠子组会推迟到第一次 collective
时才初始化。不同 pipeline stage 的第一项工作并不相同，因此各 rank 可能按不同
顺序初始化 communicator，最终互相等待。

修复位于
[`DistributedRuntime.new_group`](../src/nano_megatron/distributed/runtime.py#L225)：

- 只有 CUDA + NCCL 子组才考虑传 `device_id`；
- 通过函数签名检查确认当前 PyTorch 支持该参数；
- Gloo 和旧版 `new_group` 保持原调用方式。

[`test_distributed_runtime.py`](../tests/unit/test_distributed_runtime.py#L47) 覆盖了 NCCL
绑定、Gloo 不绑定和旧签名兼容三种情况。

### 3.6 NCCL 预热顺序遗漏模型通信组

P2P communicator 已经有预热逻辑，但旧逻辑只预热 source-colored transport
groups。仍然可能出现这样的交叉等待：一个 pipeline stage 首先进入 TP、CP、EP 或
FSDP collective，另一个 stage 却先进入 P2P receive；两边都在懒初始化不同的 NCCL
communicator。

修复后的
[`_ensure_transport_ready`](../src/nano_megatron/pipeline_parallel/p2p.py#L363)
使用一个很小的 `all_reduce`，按确定顺序先初始化 TP、CP、EP、dense replica 和
expert replica 组，再初始化 P2P transport groups，并按底层 process group 去重。

测试
[`test_nccl_warmup_initializes_model_groups_before_transports`](../tests/unit/test_pipeline_p2p.py#L183)
固定了“模型组在前、传输组在后”的顺序，同时验证重复调用不会再次预热。

### 3.7 FSDP2 API 的导入位置随 PyTorch 版本变化

旧实现只从 `torch.distributed.fsdp` 导入 `fully_shard`、`CPUOffloadPolicy` 和
`MixedPrecisionPolicy`。这个公开入口是 PyTorch 2.6 才提供的；在 2.4/2.5 中，
同一套 composable FSDP2 API 位于 `torch.distributed._composable.fsdp`。结果是代码
会把“导入路径不同”误判成“FSDP2 不存在”。

修复在 [`zero3.py`](../src/nano_megatron/data_parallel/zero3.py#L97) 中先尝试公开入口，
失败后回退到 composable 实现入口。测试辅助函数也按相同规则选择模块，使两种
PyTorch 布局都能执行 ZeRO-3 配置测试。

需要注意：当前 `pyproject.toml` 仍声明 `torch>=2.6`。因此这里描述的是源码层面的
兼容能力，不代表项目元数据正式承诺支持 2.4/2.5。

### 3.8 DCP 的 `no_dist` 参数存在版本差异

checkpoint manager 会调用 PyTorch Distributed Checkpoint 的 `save`、`load` 和
`async_save`。旧实现无条件传入 `no_dist`，但部分版本的 DCP 函数签名没有这个
可选参数，调用会直接报 `TypeError: unexpected keyword argument 'no_dist'`。

修复增加了
[`_call_dcp`](../src/nano_megatron/checkpoint/manager.py#L33)：先检查目标函数签名；
如果函数既不声明 `no_dist`，也不接受任意关键字参数，就只删除这个不受支持的
可选参数，再执行原调用。这样不会吞掉其他真正的参数错误。

测试
[`test_dcp_compatibility_drops_unsupported_no_dist_keyword`](../tests/unit/test_checkpoint.py#L51)
用旧式函数签名验证了兼容行为。

## 4. Planning 记录中出现的流程问题

planning-with-files 记录还明确写下了几类“开发流程 Bug”。它们没有改变
nano-megatron 的运行时行为，但值得保留，因为它们说明了为什么计划和验证文件也
需要机器可检查的格式。

1. 完成检查脚本最初报告 `0/0 phases complete`。原因是计划只用表格记录状态，脚本
   只识别 `### Phase` 和 `**Status:** complete`。解决方式是在计划中补充兼容的完成
   清单。
2. 第二次检查报告 `5/6`。原因是脚本使用了过宽的文本匹配，把错误日志中引用的
   状态标记也当成一个 phase。解决方式是移除会触发误计数的字面文本。
3. 一次 planning 日志补丁因为表格分隔行与预期上下文不同而失败。解决方式是先读取
   精确行，再缩小补丁上下文。
4. 一次多文件补丁遗漏了第二个文件头。解决方式是把每个文件的 patch scope 写完整，
   避免补丁内容落到错误文件。

这些问题共同指向一个原则：计划文件既是给人看的文档，也是工具消费的数据格式；
状态标记、日志文本和自动检查规则必须避免歧义。

## 5. 哪些现象没有被算作已修复 Bug

架构梳理还发现了一些看起来可疑、但没有证据表明是已修复 Bug 的现象：

- `config`、`parallel`、`distributed` 之间存在低层导入环，但项目通过纯叶子模块和
  lazy facade 控制导入顺序，这是有意设计；
- 复盘发生时 `training.metrics` 尚未接入 `Trainer`，这在当时是功能边界，不属于该批
  Bug；当前训练器已经使用 token-weighted metrics；
- `docs/architecture.md` 描述了一些尚未实现的建议目录，它是设计文档，不应被当作
  当前源码清单。

把这些内容排除在 Bug 数量之外，可以避免把“设计取舍”“未实现功能”和“真实回归”
混在一起。

## 6. 验证结果与总体经验

本次复盘运行了与上述修复直接相关的定向单元测试，结果为 `10 passed`。覆盖内容
包括后端设备选择、Tensor-only stage 输出、batched P2P、NCCL 预热顺序、NCCL
子组 `device_id`、FSDP2 配置和 DCP 旧签名兼容。

本次没有运行需要真实 CUDA/NCCL 多卡环境的 GPU 测试，因此对“实际多卡不再卡死”
的判断同时依赖修复提交、源码中的 communicator 顺序约束和已有 GPU 测试套件。

从这些 Bug 可以归纳出四条对分布式训练代码尤其重要的经验：

1. collective 的正确性不只取决于 ranks，还取决于 backend、Tensor device 和所有
   rank 的首次调用顺序；
2. 跨 FSDP、流水线和 autograd 边界的数据结构应尽量收窄为 Tensor；
3. 通信 API 应统一生命周期，避免阻塞式与异步式路径各自维护一套语义；
4. 对 PyTorch 这类快速演进的底层依赖，兼容代码要检查真实函数签名和命名空间，
   不能只依赖版本号或单一导入路径。
