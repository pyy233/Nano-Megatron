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
- 尚不支持 attention dropout 下的 CP、document-aware packed sequence 或 document
  mask；`data.packed_sequences=true` 会被拒绝，且 `CP>1` 要求 `model.dropout=0`。
- FP32 compute 只允许配 FP32 params；FP16 params 只支持拥有 FP32 master shard 的
  ZeRO-1/2，DDP 与 ZeRO-3 会拒绝。FP16 compute 当前没有 GradScaler/动态 loss scaling；
  vocab-parallel cross entropy 会把 FP16/BF16 logits 提升到 FP32 计算。
- `StepOutput.metrics` 当前是 rank-local 指标；Trainer 尚未自动跨 DP/EP/CP 做全局
  scalar reduce。

## Tokenizer 与真实文本

Tokenizer 使用独立的 Hugging Face `tokenizers` Rust 实现，不依赖 Transformers 或
Datasets。下面的命令会在 CPU 上训练 byte-level BPE，并把 `tokenizer.json` 和带
special-token/fingerprint 的 metadata 写入同一 artifact 目录：

```bash
uv run nano-megatron-tokenizer train \
  --input examples/data/tokenizer_demo.jsonl \
  --output /tmp/nano-megatron-tokenizer-demo \
  --vocab-size 300 \
  --min-frequency 1
```

这条 quickstart 在 fresh clone 中即可运行。下面的 TinyStories 命令使用本地开发
语料；`data/` 被 Git 忽略，不随仓库分发。当前核验的数据源是 ModelScope
`AI-ModelScope/TinyStories@d71d0182cc67962186d395b75b2c180340904e00`，开发子集
应准备为 `data/raw/tinystories/train_2000.jsonl`。

```bash
uv run nano-megatron-tokenizer train \
  --input data/raw/tinystories/train_2000.jsonl \
  --output data/tokenizers/tinystories-4k \
  --vocab-size 4096 \
  --min-frequency 2
```

可以直接检查 Unicode、special token 和 round-trip：

```bash
uv run nano-megatron-tokenizer inspect \
  --tokenizer data/tokenizers/tinystories-4k \
  --text 'Once upon a time, 小猫 said hello 👋' \
  --add-eos
```

小语料可以在训练启动时从 JSONL 确定性编码。更推荐先做离线预处理，避免每个
DP/EP data source rank 重复编码原文。默认的 `pt` 格式适合小语料：

```bash
uv run nano-megatron-tokenizer preprocess \
  --input data/raw/tinystories/train_2000.jsonl \
  --tokenizer data/tokenizers/tinystories-4k \
  --output data/processed/tinystories-4k.pt
```

更大的语料可以直接流式写成单 token 文件 mmap；预处理不会构造完整 token Tensor：

```bash
uv run nano-megatron-tokenizer preprocess \
  --input data/raw/tinystories/train_2000.jsonl \
  --tokenizer data/tokenizers/tinystories-4k \
  --format mmap \
  --output data/processed/tinystories-4k.mmap
```

离线 corpus 的训练配置如下；`model.vocab_size` 必须等于 `train` 命令实际报告的
`vocab_size`，TP padding 仍由模型内部处理：

```yaml
model:
  vocab_size: 4096

data:
  path: data/processed/tinystories-4k.pt
  mmap_path: null
  text_path: null
  tokenizer:
    path: data/tokenizers/tinystories-4k
    append_eos: true
  text_key: text
  num_workers: 0
  shuffle: true
  packed_sequences: false
```

若使用 mmap，将 `path` 设为 `null`，并把 `mmap_path` 指向上面的 artifact 目录；
若跳过离线预处理，则把 `path`、`mmap_path` 都设为 `null`，再令 `text_path` 指向
JSONL。三种输入都只产生满长 `input_ids/labels`；文档间插入 EOS 后按普通 causal
token stream 训练，不提供 document mask、padding mask 或 position reset。

```bash
uv run nano-megatron-train \
  --config examples/configs/gpt_tinystories_cpu.yaml \
  --max-steps 1
```

`pt` 版 `TokenCorpus` 会把完整 token stream 放入 CPU 内存。mmap artifact 则包含一个
连续的 `tokens.bin`、一个 `document_offsets.bin` 和 `metadata.json`：token payload 使用
little-endian `uint16`/`uint32` 只读映射，窗口取样时才复制一个 `S+1` block 为
`torch.long`。这里的“单文件”指唯一 token payload；offset 和校验 metadata 是必要的
sidecar。它尚未做分片，因此超大或多节点语料后续仍适合升级为 sharded mmap/indexed
corpus。`.pt` 与 mmap 对同一语料产生相同的逻辑 fingerprint，checkpoint 恢复还会绑定
sequence length、tokenizer、shuffle 和 seed；旧版 stride=`S+1` checkpoint 不能与当前
标准 stride=`S` 语义混用。
