# Megatron-LM 做了哪些性能优化

这份笔记对照 Megatron Core `779c5b748dbc` 和当前的 Nano-Megatron，只讨论训练吞吐、显存占用和多卡扩展。模型类型、平台接入和推理服务不在比较范围内。

Megatron-LM 比 Nano 快，原因并不神秘。两边用的并行数学大体相同，差距主要出在执行方式：Megatron 把小算子合并起来，把通信塞进计算空隙，用专门的 kernel 处理 attention 和 MoE，还为低精度训练、显存回收和 pipeline schedule 准备了配套实现。单看其中一个开关，收益未必很大；这些优化一起启用后，差距才会拉开。

Nano 保留了 TP、PP、DP、CP 等并行语义，但多数路径仍是普通 PyTorch 实现。这是项目目前的边界。Transformer Engine 已经移出目标，不会为了追逐 Megatron 的峰值吞吐重新接入。

## 源码从哪里看

| 内容 | Megatron-LM 源码位置 |
|---|---|
| 融合算子 | `megatron/core/fusions/`、`megatron/core/transformer/` |
| TP 和 sequence parallel | `megatron/core/tensor_parallel/`、`megatron/core/transformer/` |
| DDP 与梯度 bucket | `megatron/core/distributed/distributed_data_parallel.py`、`param_and_grad_buffer.py` |
| Distributed optimizer | `megatron/core/optimizer/distrib_optimizer.py` |
| PP schedule 和 P2P | `megatron/core/pipeline_parallel/schedules.py`、`p2p_communication.py` |
| Activation offload | `megatron/core/pipeline_parallel/fine_grained_activation_offload.py`、`megatron/core/optimizer/cpu_offloading/` |
| CP attention | `megatron/core/transformer/dot_product_attention.py`、`attention.py` |
| MoE 和 EP | `megatron/core/transformer/moe/` |
| CUDA Graph | `megatron/core/full_cuda_graph.py`、`megatron/core/transformer/cuda_graphs.py` |
| 分布式 checkpoint | `megatron/core/dist_checkpointing/` |
| 在线 resharding | `megatron/core/resharding/` |

这些目录展示的是 Megatron 能走的路径，不是默认配置。具体能否启用，还要看 GPU、CUDA、Transformer Engine、FlashAttention 等依赖和配置之间是否兼容。

## 差异概览

| 部分 | Megatron-LM | Nano-Megatron |
|---|---|---|
| 算子与 GEMM | fused bias/activation/dropout/norm、grouped GEMM、fused QKV/RoPE | 普通 PyTorch 算子 |
| Attention | FlashAttention、TE/cuDNN fused attention、packed sequence | PyTorch SDPA；支持 MHA/GQA |
| TP | 异步 AG/RS、GEMM/通信重叠、gradient accumulation fusion | 切分与 collective 已实现，重叠较少 |
| PP | 1F1B、virtual pipeline、P2P overlap、output deallocation、灵活 layout | GPipe、1F1B、interleaved 和 P2P overlap 已实现 |
| DP/ZeRO | 连续 bucket、异步梯度同步、参数 all-gather overlap、FusedAdam | DDP、ZeRO-1/2/3 和 bucketed grad 可用 |
| CP | ring、all-gather、A2A、hierarchical CP，通信与 attention 重叠 | all-gather/ring 的可读参考实现 |
| EP/MoE | fused dispatch、DeepEP、grouped expert GEMM、shared-expert overlap | 有 EP 拓扑，没有可训练的 MoE |
| 精度 | BF16/FP16、FP32 master、动态 loss scaling、FP8/FP4 | 主要是 BF16/FP32；FP16 路径较窄 |
| 显存 | selective recompute、activation offload、buffer 复用 | 有 activation checkpoint |
| CUDA Graph | 训练、验证和推理的 graph capture | 未实现 |
| 数据 | indexed dataset、document packing、变长 packed attention | JSONL、`.pt`、mmap 和确定性采样 |
| Checkpoint | fully parallel save/load、异步 I/O、跨拓扑 reshard | PyTorch DCP，支持有限的 DP resize |

## 1. 算子融合

普通 PyTorch 实现往往会把 bias、激活函数、dropout 和 residual add 分成几次 kernel launch。每一步都要读写显存，还会产生短命的中间 tensor。Megatron 则提供 fused bias-GeLU、GeGLU/SwiGLU、bias-dropout-add、fused norm、softmax、RoPE 和 vocab-parallel cross entropy。MoE 还有 fused permute、activation、unpermute 以及 grouped GEMM。

融合主要省两样东西：kernel launch 和显存带宽。它不会减少模型本身的 FLOPs，但能少搬几次数据。microbatch 较小或网络里碎算子很多时，这类开销尤其明显。

Nano 目前直接调用 PyTorch 算子。代码里能清楚看到每一步输入和输出，代价是多了 launch、临时 tensor 和显存往返。

## 2. Attention

Megatron 可以把 attention 交给 FlashAttention、Transformer Engine fused attention 或 cuDNN attention。这些实现按块计算 softmax，不需要把完整的 `S × S` attention matrix 留在显存里。长序列时，它们通常同时改善速度和峰值显存。

Megatron 还支持 THD/packed sequence。多篇长度不同的文档可以压进同一条 token stream，kernel 通过 `cu_seqlens` 识别边界，不必为最短文档补到最长长度。QKV projection、RoPE 和输出投影也有融合或 buffer 复用路径。推理侧另有 flash decode、fused KV append 等实现，不过那不是本文关注的重点。

Nano 使用 PyTorch SDPA。数值和 shape 比较容易检查，但它不是 Megatron 的 fused-attention 路径。进入 CP 后，Nano 也没有把 K/V 通信和 attention 计算做成完整的异步流水线；packed sequence、document mask 和 attention dropout 目前都不支持。

## 3. Tensor Parallel

切分线性层只是 TP 的第一步。真正难的是不让 all-gather 和 reduce-scatter 把 GPU 晾在一边。Megatron 为此做了几层处理：

- sequence parallel 把部分 activation 沿 sequence 维分开，减少每张卡保存的内容；
- all-gather/reduce-scatter 可以异步发起，与 GEMM 或后面的计算重叠；
- user buffer 和预先分配的通信 buffer 减少 copy 和同步；
- gradient accumulation fusion 让多个 microbatch 的梯度直接累加到固定 buffer；
- 部分 weight-gradient GEMM 可以延迟到 pipeline drain 阶段；
- vocab-parallel embedding、输出层和 cross entropy 避免在每张卡上复制完整 logits。

rank order 也会影响速度。Megatron 会尽量把 TP、CP、EP 这些高频通信组放在 NVLink 或 NVSwitch 范围内，跨节点的低带宽链路留给频率较低的通信。

Nano 已有 TP、sequence parallel 和 vocab-parallel loss，collective 的语义也是正确的。不过主路径多半要等通信完成再继续算，能隐藏的通信时间远少于 Megatron。

## 4. Pipeline Parallel

Megatron 的 PP schedule 覆盖非交错 1F1B、interleaved/virtual pipeline，以及 warmup、steady-state、cooldown 各阶段。virtual pipeline 会把一个物理 stage 再拆成几个 chunk，用更细的流水减小 bubble。P2P send/recv 可以批量或异步提交，尽量与前后向计算同时进行。

它还会处理一些不太显眼的内存问题。比如 forward 输出发给下一 stage 后，可以释放大块 storage，只留下 backward 所需的信息；embedding 的 weight-gradient GEMM 也能延后执行。非均匀 pipeline layout 则允许按实际计算量分层，而不是机械地给每个 stage 相同层数。

Nano 这一块不是空白：GPipe、1F1B、interleaved 1F1B、P2P overlap 和动态 activation shape 协议都已实现。缺的是 output deallocation、embedding WGrad deferral、自动或灵活的非均匀 layout，以及 Megatron 中那些和 MoE、offload、CUDA Graph 交织在一起的 schedule 分支。

## 5. Data Parallel、梯度同步和 ZeRO

Megatron DDP 把参数和梯度整理到连续 buffer，再按 dtype、dense/expert 参数域和 bucket 切分。一个 bucket 的梯度全部就绪后，reduce 或 reduce-scatter 会立即启动，不必等整个 backward 结束。这样，前面层的通信可以躲在后面层的反向计算下面。

Distributed optimizer 继续把 optimizer state、gradient 和 parameter 分片，并管理 FP32 main parameter 与低精度 model parameter 的拷贝。参数 all-gather 可以提前发起，与下一次 forward 重叠。生产路径还接入了 FusedAdam、多个 optimizer instance、CPU offload、量化通信和多种 checkpoint 格式。

Dense 参数和 expert 参数不能随手共用同一个 DP group。Megatron 为两者维护不同的 buffer、replica group 和梯度缩放规则，这一点对 MoE 的正确性也很重要。

Nano 已实现 DDP、ZeRO-1/2/3、bucketed gradient 和显式 parameter domain。差距主要在参数 all-gather/forward overlap、fused optimizer、量化通信、Megatron FSDP 和 distributed-optimizer checkpoint。这些功能不是补几个配置字段就能得到的，它们会改动 backward hook、buffer 生命周期和每一步的同步位置。

## 6. Context Parallel

Megatron 的 CP 可选 all-gather、ring/P2P、A2A 和 hybrid/hierarchical 方案。不同算法适合不同节点拓扑和 attention 形式。高性能路径会一边传 K/V block，一边计算当前 block 的 QK、softmax 和 AV，而不是等完整 context 到齐再开始。

这套流水依赖专用 attention kernel、online softmax 和正确的 RNG offset。variable-length sequence、sequence parallel、dropout 和 backward 都要遵守同一套全局坐标，否则不同 CP size 会得到不同结果。

Nano 的 all-gather CP 和 ring CP 是 correctness reference。ring backward 的梯度语义已经验证，但通信结束后才进入普通 PyTorch matmul。zigzag/hierarchical CP、packed attention 和 CP dropout 还没有接入。

## 7. Expert Parallel 和 MoE

MoE 的瓶颈往往不是 router 本身，而是 token 重排、all-to-all 和许多大小不一的 expert GEMM。Megatron 在这几处都有专门实现：

- all-gather、all-to-all、Flex dispatcher，以及 DeepEP/NVSHMEM 等通信后端；
- fused permute/unpermute 和 routing map；
- grouped GEMM，一次处理多个本地 expert；
- expert TP、expert DP 和 shared expert；
- shared expert 计算与 EP 通信重叠；
- capacity、token drop/padding、router aux loss 和 z-loss。

这些路径还要和 1F1B、activation recompute、CUDA Graph 以及低精度训练配合，代码量自然会膨胀。

Nano 当前有 EP topology，也区分 dense/expert replica group，但没有 dispatcher、expert MLP 或 token routing。换句话说，`EP=2` 的组合测试只验证了拓扑和副本语义，不能据此说 Nano 已经支持 Megatron 风格的 MoE 训练。

## 8. 低精度训练

Megatron 的 FP16 路径有 FP32 master parameter、动态 loss scaling、overflow 检测和 gradient unscale。FP8 还需要 amax history、scaling recipe、不同 tensor 的量化策略以及配套 GEMM。新版本又加入了面向 Hopper/Blackwell 的 FP8、FP4/NVFP4、量化参数 all-gather 和 grouped GEMM。

这不是把参数转换成低精度 dtype 就算完成。GEMM 输入、通信格式、master weight、optimizer state 和 checkpoint 都会受影响，缺一块就可能出现性能倒退或数值问题。

Nano 主要验证 BF16/FP32。FP16 只在有限配置中可用，没有完整的 GradScaler、FP8/FP4 recipe 或量化 distributed optimizer。Transformer Engine 也不会作为当前项目的解决办法重新引入。

## 9. Activation、offload 和显存管理

Megatron 支持整层重算，也能只重算 attention、MLP 或 MoE 的指定部分。Selective recompute 通常比粗粒度 checkpoint 更容易在计算量和显存之间找到合适的位置。保存下来的 activation 还可以在模型并行 rank 间分摊。

显存仍然不够时，Megatron 可以把部分 activation 搬到 CPU。这里并不是简单调用一次 `.cpu()`：实现里有 pinned-memory pool、独立 D2H/H2D stream、CUDA event 和 inflight 数量限制，还要按 pipeline stage 平衡 offload 量。否则 PCIe 传输很容易从省显存的办法变成新的瓶颈。

Nano 有 activation checkpoint，但没有这套 module-level offload scheduler，也没有 PP-aware 的 CPU 传输和 buffer 管理。

## 10. CUDA Graph

CUDA Graph 把固定的 kernel 和 collective 序列捕获下来，后续用一次 replay 提交，省掉每个 microbatch 的 Python 调度和 kernel launch 间隙。Megatron 能捕获训练、验证和部分 optimizer 路径，也会为不同 batch 或 token bucket 保存多张 graph。推理路径则把固定 batch、KV cache 和 flash decode 一并纳入。

代价是静态约束。shape、控制流、buffer 地址、RNG 和 collective 顺序都不能随意变化，MoE dispatcher、offload 和 optimizer 也必须使用 graph-safe buffer。Megatron 为这些组合写了专门的检查和执行路径。

Nano 目前仍是 eager/autograd 执行，没有 CUDA Graph。对这个项目来说，普通执行更容易检查 schedule、RNG 和 collective；吞吐上则会留下 CPU launch gap。

## 11. 数据与序列组织

Megatron 使用 indexed dataset、mmap、索引缓存和分布式 sample mapping，尽量让 GPU 不等数据。它也支持 document-aware packing：多个短文档共享一个训练序列，attention 通过边界信息阻止文档互相看到。变长输入以 `cu_seqlens`/THD 表示，可以直接送进 FlashAttention，而不是先补齐 padding。

训练侧还有 batch-size ramp-up、microbatch calculator 和按 token 数推进进度等工具。它们未必让单个 kernel 更快，但会影响 pipeline bubble、优化器稳定性和整个训练周期的利用率。

Nano 已有流式 mmap、确定性的多 epoch 采样、validation，以及恢复时的数据位置。它还没有 document-aware packing、THD attention 输入和 Megatron 那套大规模 indexed-dataset cache。

## 12. Checkpoint 和运行时

Checkpoint 不改变训练 FLOPs，却会直接造成停顿。Megatron 的 `ShardedTensor`/`ShardedStateDict` 记录全局 shape、offset、replica 和 shard metadata；fully parallel save/load 把 rank 间通信和本地 I/O 分开调度，异步保存则允许训练继续往前跑。

这些 metadata 也用于改变并行拓扑。Megatron 可以转换 checkpoint 的 TP、PP、EP、DP layout。在 RL 场景里，`megatron/core/resharding` 还能把正在训练的权重直接送到另一套 inference layout，不必先落盘再加载。底层另有 topology-aware communicator、NCCL allocator、NVSHMEM/UCC 等后端。

Nano 使用 PyTorch DCP 和自己的 shard metadata，已经覆盖正常保存、恢复和有限的 DP resize。任意 TP/PP/EP 变更、fully parallel checkpoint、在线 resharding 和多通信后端还不在当前实现里。

## 哪些优化最值得关注

同一模型、batch、精度和拓扑下，最先拉开速度的通常是 fused/Flash Attention、GEMM 与逐元素算子融合。卡数增加后，TP/DP/PP/CP 的通信重叠开始主导差距。显存方面，sequence parallel、selective recompute、offload 和 distributed optimizer 更重要。

MoE 要单独看。没有 grouped GEMM、fused dispatch 和高效 all-to-all，即使 EP 的数学语义正确，吞吐也很难接近 Megatron。

Nano 的 reference 路径不是低配版 Megatron，也不该逐行模仿它。这个仓库更适合保留一条能读、能测、能和优化实现对照的路径。以后真要加性能功能，可以按下面的顺序来：

1. 用 profiler 记录 kernel、通信和显存基线；
2. 每次只加一个 fused operation，并和原始 PyTorch 路径做数值对照；
3. 先处理 TP/DP overlap，再碰 CP/EP overlap；
4. 接入 attention backend 时保留 SDPA fallback；
5. 最后评估 activation offload、FP8/FP4、CUDA Graph 和 MoE fused dispatch。

每个新路径至少要有单卡数值回归、一个多卡 collective 测试，以及关闭优化后仍可运行的 reference 实现。
