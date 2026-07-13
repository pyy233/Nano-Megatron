# Nano-Megatron

一个以可读性和可验证性为优先的 mini Megatron 教学实现。

## 当前实现边界

- 已在 CPU/Gloo 上验证 GPT、TP/SP、PP、CP、EP topology、DDP/ZeRO-1/2/3、
  梯度累积、激活重计算、offload 和 checkpoint 的主要正确性路径。测试集
  持续变化，可用 `.venv/bin/pytest -q` 获取当前结果。
- 通信 overlap 包括 DDP/ZeRO-1/2 的 bucketed gradient reduction，以及三种 PP
  schedule 的 P2P send/recv。GPipe、非交错 1F1B 和 interleaved 1F1B 共用同一个
  schedule executor 与 exchange 生命周期；ZeRO-3 的梯度通信由 FSDP2 管理。
- PP 支持 GPipe、非交错 1F1B 和 virtual/interleaved 1F1B。每个物理 PP rank
  可持有多个 model chunk；P2P 既支持固定 activation shape，也支持在每个
  microbatch 传输 shape metadata 的动态协议。
- checkpoint 支持同步和异步保存，`checkpoint.async_save` 默认为 `false`。异步
  路径先生成与后续训练解耦的 CPU snapshot，只在所有 rank 写入成功后
  发布完整 checkpoint；ZeRO-3 复用 PyTorch DCP 的 `async_save`。
- 已在 2×A40/NCCL 上完成 DDP/ZeRO-1/2 gradient overlap、PP2/VP2 交错
  pipeline + P2P overlap + dynamic shape，以及 ZeRO-3 异步 checkpoint
  保存/恢复、ZeRO-3 参数/梯度 CPU offload 的定向验证。真实 Transformer
  Engine 和需要四卡以上的多维并行组合仍需要后续实机覆盖。
- 多进程 DDP/ZeRO-1/2 checkpoint 使用共享文件系统上的 rank-local shard，支持 dense
  GPT 沿 DP/EP replica 轴恢复。ZeRO-3 使用 FSDP2 canonical model/optimizer state + DCP，
  dense GPT 按 `(TP, PP)` 保存独立子目录；恢复要求 TP/PP/CP/EP 不变，支持改变 DP degree。
- CP 不支持 attention dropout、packed sequence 或 document mask；当前集中校验要求
  `CP>1` 时 `model.dropout=0`。
- FP32 compute 只允许配 FP32 params；FP16 params 只支持拥有 FP32 master shard 的
  ZeRO-1/2，DDP 与 ZeRO-3 会拒绝。FP16 compute 当前没有 GradScaler/动态 loss scaling；
  vocab-parallel cross entropy 会把 FP16/BF16 logits 提升到 FP32 计算。
- `StepOutput.metrics` 当前是 rank-local 指标；Trainer 尚未自动跨 DP/EP/CP 做全局
  scalar reduce。
