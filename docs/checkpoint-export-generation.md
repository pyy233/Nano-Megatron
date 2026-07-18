# Checkpoint 导出与单卡生成

## 1. 目标

训练 checkpoint 服务于精确续训，包含 model、optimizer、trainer、RNG 和 topology 状态；
推理 artifact 只携带完整单卡 GPT 权重、模型配置和经过 fingerprint 校验的 tokenizer。

```text
TP/PP/DP training checkpoint
→ manifest-driven TP merge
→ PP layer remap
→ model-only single-device artifact
→ CPU / one CUDA GPU autoregressive generation
```

## 2. 导出

```bash
uv run nano-megatron-export \
  --checkpoint checkpoints/<run>/<timestamp>/step_XXXXXXXX \
  --output exported/<name>
```

默认使用 checkpoint manifest 内嵌的 `run_config`，并从其中寻找训练 tokenizer。如果旧或
程序化 checkpoint 没有完整 provenance，可以显式指定：

```bash
uv run nano-megatron-export \
  --checkpoint checkpoints/<run>/<timestamp>/step_XXXXXXXX \
  --config examples/configs/<training-config>.yaml \
  --tokenizer data/tokenizers/<tokenizer> \
  --output exported/<name>
```

显式配置不能改变 checkpoint 的模型结构或 TP/PP/CP/EP/DP sizes；tokenizer vocab 必须
精确等于 `model.vocab_size`。

导出过程：

1. 要求 source checkpoint 有 `.complete` 和 schema-compatible `manifest.json`；
2. 每次只读取 `(TP, PP, EP)` 唯一 writer 的 model state，忽略 optimizer；
3. 按 `ShardMetadata.global_offset/local_shape` 放置 TP shards，不从参数名猜 concat dim；
4. 按 PP `layer_start/layer_end` 把 stage-local layer index 改为全局 layer index；
5. 对 TP replicated tensors 逐值比较，对 tied embedding/head 要求完全一致；
6. 检查完整 state keys/shapes 与 TP1/PP1 GPT 一致；
7. 复制并重新加载 tokenizer，写入权重 SHA256 和 tokenizer fingerprint；
8. 在临时目录完整自校验后原子发布，拒绝覆盖已有非空目录。

支持矩阵：

| Source checkpoint | 首版支持 |
|---|---|
| 多进程 DDP rank-local | 是 |
| 多进程 ZeRO-1/2 rank-local | 是，model layout 与 DDP 相同 |
| 单进程 `torch.save` model state | 是 |
| ZeRO-3 FSDP2 DCP | 否 |
| `virtual_stages_per_rank > 1` | 否 |
| dense EP1 | 是 |
| EP>1 / expert parameters | 否 |

导出 artifact：

```text
<output>/
├── .complete
├── manifest.json
├── model.pt
└── tokenizer/
    ├── metadata.json
    └── tokenizer.json
```

`manifest.json` 记录 source checkpoint step/backend/topology/manifest SHA256、完整 GPT 配置、
权重文件 SHA256/dtype/tensor count，以及 tokenizer vocab/fingerprint/EOS id。加载时会重新
验证所有字段。

## 3. 单卡生成

```bash
CUDA_VISIBLE_DEVICES=0 uv run nano-megatron-generate \
  --model exported/<name> \
  --prompt 'Once upon a time, there was a little fox' \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 0 \
  --seed 1234
```

默认行为：

- `--device auto`：有 CUDA 时用 CUDA，否则 CPU；
- `--dtype auto`：CUDA 保留 artifact dtype，CPU 提升为 FP32；
- 默认添加 BOS，不预加 EOS；生成到 EOS 时停止；
- `temperature=0` 为 greedy，否则在 CPU 上执行可复现的 top-k/top-p sampling；
- prompt 与 generation 的总 token 数必须不超过训练时 `model.seq_length`；
- 默认输出完整文本，`--json` 输出 prompt/completion/token ids/stop reason。

固定 seed 只约束 sampling；同一 artifact、prompt 和参数会产生相同 token 序列。

## 4. 当前性能边界

首版没有 KV cache。每一步都会把完整 prefix 重新送入模型，因此计算量随生成长度快速增长。
这条路径适合：

- checkpoint 正确性与 logits 验收；
- TinyStories 固定 prompts 的质量检查；
- 小 batch 本地演示。

它不适合高吞吐在线 serving。后续可以在不改变 artifact 契约的情况下，为 attention/RoPE
增加 per-layer KV cache，或者再实现 Hugging Face/vLLM export。

## 5. 已验证证据

- 单进程 `torch.save` checkpoint→artifact→CPU greedy generation；
- artifact collision、权重 SHA256 篡改和 context overflow 拒绝；
- 真实 8-process Gloo `TP2×PP2×DP2` checkpoint 只写 4 个 model shards；
- 真实 8-process Gloo `TP2×PP2×DP2` 的 ZeRO-1/2 rank-local checkpoint 均可导出；
- 导出后单卡完整 logits 与分布式 TP-gather logits 在 `atol=rtol=2e-5` 内一致；
- 导出 artifact 可继续完成单卡 generation smoke。
