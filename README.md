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
  发布完整 checkpoint；ZeRO-3 复用 PyTorch DCP 的 `async_save`。CLI 将
  `checkpoint.directory` 视为 base，每次启动在其下创建
  `YYYYMMDD-HHMMSS-ffffff/step_*`；时间目录由 rank 0 分配并同步给所有 rank。
- 已在 2×A40/NCCL 上完成 DDP/ZeRO-1/2 gradient overlap、PP2/VP2 交错
  pipeline + P2P overlap + dynamic shape，以及 ZeRO-3 异步 checkpoint
  保存/恢复、ZeRO-3 参数/梯度 CPU offload 的定向验证。另已在 8×A100/NCCL 上完成
  `TP2×PP2×CP2`、`TP2×PP2×DP2`、`TP2×CP2×DP2`、`TP2×CP2×EP2` 和
  `CP2×EP2×DP2` 的 30-step BF16 短收敛、validation、checkpoint/load 验证；最复杂的
  `TP2×PP2×CP2` 还完成 step 15→30 精确续训。这里 EP 在 dense GPT 中仍表示额外
  batch replica，不是 MoE expert dispatch；完整 TinyStories 长收敛仍待执行。
- 同一台 8×A100 还完成了扩展矩阵：不设 `NCCL_P2P_DISABLE` 的原生 NCCL
  transport、ring CP、PP P2P overlap、VP2/interleaved 1F1B，以及开启
  dynamic activation-shape metadata 协议的 10-step BF16 训练/验证均通过。
  ZeRO-1/2/3 与上述五个三维拓扑的 15 组 10-step 笛卡尔积也全部通过，
  并在 `TP2×PP2×DP2` 上分别完成 ZeRO-1/2/3 的 step 5 保存→新进程
  step 6 恢复；ZeRO-3 还验证了 async save 发布 `.complete`。dynamic 协议
  本次的数据仍固定为 seq=64，因此是协议路径验证，不是变长 microbatch
  矩阵。这些均是短程 correctness smoke，不等于完整 TinyStories 收敛。
- 多进程 DDP/ZeRO-1/2 checkpoint 使用共享文件系统上的 rank-local shard，支持 dense
  GPT 沿 DP/EP replica 轴恢复。ZeRO-3 使用 FSDP2 canonical model/optimizer state + DCP，
  dense GPT 按 `(TP, PP)` 保存独立子目录；恢复要求 TP/PP/CP/EP 不变，支持改变 DP degree。
- 尚不支持 attention dropout 下的 CP、document-aware packed sequence 或 document
  mask；`data.packed_sequences=true` 会被拒绝，且 `CP>1` 要求 `model.dropout=0`。
- 311M TinyStories 正式配置已在一台 8×L40S RunPod 上完成 batch sweep、500-step 长稳、
  step500→600 精确恢复和真实 checkpoint 单卡生成。该节点必须设置
  `NCCL_P2P_DISABLE=1`；实测结果、磁盘约束及正式启动/恢复命令见
  [`docs/tinystories-8x-l40s-runbook.md`](docs/tinystories-8x-l40s-runbook.md)。
- FP32 compute 只允许配 FP32 params；FP16 params 只支持拥有 FP32 master shard 的
  ZeRO-1/2，DDP 与 ZeRO-3 会拒绝。FP16 compute 当前没有 GradScaler/动态 loss scaling；
  vocab-parallel cross entropy 会把 FP16/BF16 logits 提升到 FP32 计算。
- Trainer 已接入 warmup + constant/cosine LR、确定性多 epoch、周期性 validation、
  token-weighted loss/perplexity、吞吐日志和可选 W&B。loss 在最后 PP stage 上跨
  `DP×EP×CP` 汇总，再沿 PP 路由；TP 不重复计数。
- DDP/ZeRO-1/2 的 rank-local TP/PP checkpoint 可以离线合并成自包含单卡 artifact；
  artifact 绑定模型配置、权重 SHA256 和 tokenizer fingerprint，并由
  `nano-megatron-generate` 在 CPU 或单张 CUDA GPU 上生成。首版仅支持 dense EP1、VP1，
  尚不支持 ZeRO-3 DCP export、HF/vLLM 格式或 KV cache。

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

这条 quickstart 在 fresh clone 中即可运行。正式 TinyStories tokenizer 使用仓库脚本
从被删除或不存在的 `data/` 开始重建。下载脚本固定数据 revision、文件大小和 SHA256，
支持 `.part` 断点续传，并从固定 TXT 的 2,119,489 篇非空故事中确定性抽取严格
500,000 篇（原文件另有 230 个空白 record，会显式排除）：

```bash
uv run python scripts/download_tinystories.py

uv run python scripts/train_tinystories_tokenizer.py \
  --threads 16
```

输出分别是 `data/raw/tinystories/train_500k.jsonl` 和
`data/tokenizers/tinystories-8k-500k/`。训练脚本只有在重新加载后实际词表恰好为
8192、输入文档数恰好为 500,000 时才发布 artifact；重复执行会验证并复用完整结果。
完整的下载固定值、抽样算法、磁盘/内存需求和故障恢复说明见
[`docs/tinystories-tokenizer.md`](docs/tinystories-tokenizer.md)。

可以直接检查 Unicode、special token 和 round-trip：

```bash
uv run nano-megatron-tokenizer inspect \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --text 'Once upon a time, 小猫 said hello 👋' \
  --add-eos
```

不要用同一份 token stream 同时训练和验证。若先用 500k tokenizer sample
做本地 GPT 开发训练，可以做确定性的 90/10 哈希拆分；同文文本（包括重复
样本）始终进入同一侧：

```bash
uv run nano-megatron-tokenizer split \
  --input data/raw/tinystories/train_500k.jsonl \
  --train-output data/raw/tinystories/train_split.jsonl \
  --validation-output data/raw/tinystories/validation_split.jsonl \
  --validation-fraction 0.1 \
  --seed 1234
```

小语料可以在训练启动时从 JSONL 确定性编码。更推荐先做离线预处理，避免每个
DP/EP data source rank 重复编码原文。训练集和验证集使用同一个 tokenizer artifact，
但分别编码为互斥的 token corpus。默认的 `pt` 格式适合小语料：

```bash
uv run nano-megatron-tokenizer preprocess \
  --input data/raw/tinystories/train_split.jsonl \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --output data/processed/tinystories-8k-500k-train.pt

uv run nano-megatron-tokenizer preprocess \
  --input data/raw/tinystories/validation_split.jsonl \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --output data/processed/tinystories-8k-500k-validation.pt
```

更大的语料可以直接流式写成单 token 文件 mmap；预处理不会构造完整 token Tensor：

```bash
uv run nano-megatron-tokenizer preprocess \
  --input data/raw/tinystories/train_split.jsonl \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --format mmap \
  --output data/processed/tinystories-8k-500k-train.mmap
```

离线 corpus 的训练配置如下；`model.vocab_size` 必须等于 `train` 命令实际报告的
`vocab_size`，TP padding 仍由模型内部处理：

```yaml
model:
  vocab_size: 8192

data:
  path: data/processed/tinystories-8k-500k-train.pt
  mmap_path: null
  text_path: null
  tokenizer:
    path: data/tokenizers/tinystories-8k-500k
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
  --max-steps 5
```

训练配置可直接启用完整闭环：

```yaml
lr_scheduler:
  schedule: cosine
  warmup_steps: 100
  decay_steps: 10000       # null 时使用 training.max_steps
  min_lr: 0.00003

validation:
  interval: 5
  batches: 10
  data:
    path: data/processed/tinystories-8k-500k-validation.pt
    mmap_path: null
    text_path: null
    text_key: text
    tokenizer:
      path: data/tokenizers/tinystories-8k-500k
      append_eos: true
    num_workers: 0
    shuffle: false
    packed_sequences: false

wandb:
  enabled: false
  project: nano-megatron
  mode: online             # online / offline / disabled
```

W&B 是可选依赖；需要时先执行 `uv sync --extra tracking`，再把 `wandb.enabled` 设为
`true`。只有全局主 rank 会创建 run。训练/验证 loss、perplexity、grad norm、LR、吞吐、
step、累计 sample/token、epoch 和当前 epoch 的 sample offset 都会记录；run ID 也进入
checkpoint。

每个时间目录都会由框架维护一份与 W&B 无关的 `metrics.jsonl`，即使 W&B 被禁用也会
写入。文件按事件逐行记录 train/validation metrics、trainer/data 位置、版本、sequence 和
step，并在 checkpoint 中保存最后一条历史的 sequence/step。使用 `--resume` 时，新时间
目录先复制源 `metrics.jsonl` 中不超过 checkpoint step 的记录，然后 W&B 创建一个普通的
全新 run（不使用服务端私有预览的 `fork_from`），回放这些记录，再继续追加新的 step。
原 W&B run 和原 metrics 文件保持不变，新 run ID 会写入后续 checkpoint。旧 checkpoint
没有 `metrics.jsonl` 时会给出警告并从空 history 启动，不阻止模型恢复。

拆分后训练 corpus 的 fingerprint 会变化，因此以前基于其他 TinyStories corpus 保存的
checkpoint 不能续训到这份新配置；请使用新的 checkpoint 目录或删除配置中的 resume
参数，从 step 0 启动。示例配置以 `checkpoints/gpt_tinystories_cpu_split` 为 base，
实际写入 `checkpoints/gpt_tinystories_cpu_split/<时间点>/step_*`。
validation 只读取固定 held-out corpus，不参与 optimizer update，控制台和 W&B 中分别显示
`validation loss`/`perplexity` 和
`validation/loss`/`validation/perplexity`。

每次 `--resume` 也会创建新的时间目录：`--resume` 是读取路径，当前配置中的
`checkpoint.directory` 是新分支的保存 base。`checkpoint.keep_last` 只清理本次时间目录
内的旧 step，不会删除其他时间目录中的训练结果。启动日志会打印最终解析出的
`checkpoint run directory`。目录结构如下：

```text
<checkpoint base>/<时间点>/
├── metrics.jsonl
├── step_00000200/
└── step_00000400/
```

`pt` 版 `TokenCorpus` 会把完整 token stream 放入 CPU 内存。mmap artifact 则包含一个
连续的 `tokens.bin`、一个 `document_offsets.bin` 和 `metadata.json`：token payload 使用
little-endian `uint16`/`uint32` 只读映射，窗口取样时才复制一个 `S+1` block 为
`torch.long`。这里的“单文件”指唯一 token payload；offset 和校验 metadata 是必要的
sidecar。它尚未做分片，因此超大或多节点语料后续仍适合升级为 sharded mmap/indexed
corpus。`.pt` 与 mmap 对同一语料产生相同的逻辑 fingerprint，checkpoint 恢复还会绑定
sequence length、tokenizer、shuffle 和基础 seed。checkpoint 还显式保存 epoch、该
epoch 的 shuffle seed 语义和已提交的全局 sample offset；DataLoader worker 预取不会
提前推进这个 offset。旧版 stride=`S+1` checkpoint 不能与当前标准 stride=`S` 语义混用。

## Checkpoint 导出与单卡生成

训练 checkpoint 包含 optimizer、trainer 和 rank-local runtime state；推理前先把其中的
model shards 合并成独立 artifact。正式训练的 DDP、TP2×PP2×DP2 checkpoint 可以直接：

```bash
uv run nano-megatron-export \
  --checkpoint checkpoints/tinystories-a100-311m-tp2-pp2-dp2/<timestamp>/step_XXXXXXXX \
  --output exported/tinystories-311m
```

导出器默认从 checkpoint manifest 的 `run_config` 读取模型配置和 tokenizer 路径；旧
checkpoint 缺少这些信息时可显式传 `--config` 和 `--tokenizer`。目标目录拒绝覆盖并先在
同级临时目录完成权重拼接、TP replica/tied-weight 校验、SHA256 和 tokenizer 自校验，再
原子发布：

```text
exported/tinystories-311m/
├── .complete
├── manifest.json
├── model.pt
└── tokenizer/
    ├── metadata.json
    └── tokenizer.json
```

单卡生成：

```bash
CUDA_VISIBLE_DEVICES=0 uv run nano-megatron-generate \
  --model exported/tinystories-311m \
  --prompt 'Once upon a time, there was a little fox' \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 1234
```

`--temperature 0` 使用 greedy decoding；还支持 `--top-k`、`--no-add-bos`、`--json`、
`--device cpu/cuda:N` 和 `--dtype`。命令严格要求 prompt tokens 与新增 tokens 之和不超过
训练配置的 `model.seq_length`，并在 EOS 时停止。当前实现每生成一个 token 都重算完整
prefix，适合质量验收，不是高吞吐 serving；KV cache 是后续独立优化。

详细 artifact 契约、支持矩阵和故障含义见
[`docs/checkpoint-export-generation.md`](docs/checkpoint-export-generation.md)。
