# Nano-Megatron

一个以可读性和可验证性为优先的 mini Megatron 教学实现。

- [第一阶段 GPT 架构设计](docs/architecture.md)
- [Megatron-LM 源码架构对比](docs/megatron-comparison.md)

## 当前实现边界

- 已在 CPU/Gloo 上验证 GPT、TP/SP、PP、CP、EP topology、DDP/ZeRO-1/2/3、
  梯度累积、激活重计算和 checkpoint 的主要正确性路径；当前全量结果为
  `184 passed, 8 warnings in 241.95s`。8 条 warning 均为单进程 DCP 在
  `torch.distributed` 未初始化时的预期提示；Ruff 和单步 CPU CLI 也已通过。
- CUDA/NCCL、真实 Transformer Engine，以及 GPU 上的 activation/optimizer/FSDP
  CPU offload 尚未实机验证；现阶段不能把 CPU/Gloo 结果等同于 GPU 后端验收。
- 多进程 DDP/ZeRO-1/2 checkpoint 使用共享文件系统上的 rank-local shard，支持 dense
  GPT 沿 DP/EP replica 轴恢复。ZeRO-3 使用 FSDP2 canonical model/optimizer state + DCP，
  dense GPT 按 `(TP, PP)` 保存独立子目录；恢复要求 TP/PP/CP/EP 不变，支持改变 DP degree。
- CP 不支持 attention dropout、packed sequence 或 document mask；当前集中校验要求
  `CP>1` 时 `model.dropout=0`。
- PP 只支持静态 activation shape 的 GPipe 和非交错 1F1B，不支持 virtual/interleaved
  pipeline 或 P2P overlap。
- FP32 compute 只允许配 FP32 params；FP16 params 只支持拥有 FP32 master shard 的
  ZeRO-1/2，DDP 与 ZeRO-3 会拒绝。FP16 compute 当前没有 GradScaler/动态 loss scaling；
  vocab-parallel cross entropy 会把 FP16/BF16 logits 提升到 FP32 计算。
- `StepOutput.metrics` 当前是 rank-local 指标；Trainer 尚未自动跨 DP/EP/CP 做全局
  scalar reduce。
