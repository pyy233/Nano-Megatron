<div align="center">

# Nano-Megatron

**一个以可读性和可验证性为优先的 mini Megatron 分布式训练框架。**

[快速开始](#快速开始) · [功能边界](#功能边界) · [Megatron Core 对比](docs/megatron-optimizations.md) · [311M 模型与训练归档](https://huggingface.co/FishBoard/nano-megatron-tinystories-311m/tree/main)

</div>

Nano-Megatron 用尽量直接的 PyTorch 代码串起 GPT 训练中的并行组、collective、pipeline
schedule、梯度同步、checkpoint 和恢复流程。它的主要用途是学习分布式训练的基本逻辑，
不是替代 Megatron-LM。

仓库包含一条完整、可运行的训练链路：训练 byte-level BPE tokenizer，流式预处理
TinyStories，使用 TP、PP、CP、DP/ZeRO 等并行方式训练 dense GPT，记录验证指标与 W&B，
保存和恢复分布式 checkpoint，最后导出单卡模型并生成文本。

## 已验证模型

完整的 TinyStories 311M 实验已经在 8×L40S 上训练完成：

| 项目 | 结果 |
|---|---|
| 模型 | 310,821,888 参数，24 layers，hidden 1024，16 heads |
| 并行拓扑 | TP2 × PP2 × DP2，BF16，sequence length 512 |
| 训练 | 全量 TinyStories，global batch 256，3 epochs |
| 最佳验证结果 | loss 1.1795，perplexity 3.2528，step 10250 |

推理模型、完整可恢复 checkpoint、tokenizer、训练配置、metrics、W&B 原始记录、终端日志和
SHA256 清单都在：

**[FishBoard/nano-megatron-tinystories-311m](https://huggingface.co/FishBoard/nano-megatron-tinystories-311m/tree/main)**

## 功能边界

### 包含

| 模块 | 已实现 |
|---|---|
| GPT | RMSNorm、RoPE、SwiGLU、GQA/MHA、tied embeddings、PyTorch SDPA |
| 并行训练 | DDP、ZeRO-1/2/3、TP、sequence parallel、PP、CP，以及并行组拓扑中的 EP |
| Pipeline | GPipe、1F1B、interleaved 1F1B、P2P overlap、动态 activation shape 协议 |
| 训练配方 | AdamW、梯度累积、梯度裁剪、warmup、constant/cosine LR、activation checkpoint |
| 数据 | byte-level BPE、JSONL、PyTorch token corpus、流式 mmap、多 epoch 确定性采样 |
| 可观测性 | validation、loss/perplexity、吞吐、显存指标、W&B、独立的 metrics.jsonl |
| Checkpoint | 同步/异步保存、精确恢复、数据位置与 RNG 状态、best/final checkpoint |
| 推理闭环 | 合并 DDP/ZeRO-1/2 的 TP/PP shards，导出原生单卡 artifact，采样生成 |
| 验证 | CPU/Gloo 多进程测试，以及 2×A40、8×L40S 的真实 NCCL 测试 |

### 没有包含

- 没有 Transformer Engine、自定义 fused kernel 或对 FlashAttention 的直接依赖；kernel
  backend 目前只有 PyTorch。
- 没有 MoE expert dispatch。dense GPT 中的 EP 只提供拓扑与额外 replica 语义。
- 没有生产级推理服务、KV cache、continuous batching、Transformers/vLLM 格式
- 没有 document-aware packed attention、document mask 或 position reset；CP 也不支持
  attention dropout。
- 没有把多节点容错、集群调度和超大规模性能优化作为目标。

## 推荐阅读人群

这个仓库适合：

- 正在学习 PyTorch Distributed、Megatron 并行维度或 ZeRO/FSDP 的读者；
- 想从 tensor shape、process group 和 collective 层面理解训练流程的学生和研究者；
- 希望在单机多卡上修改 schedule、通信或 checkpoint，并用小模型快速验证的人；
- 读 Megatron-LM 源码时觉得工程层太厚，希望先看一个显式、可测试实现的人。

如果你需要开箱即用的最高训练吞吐、成熟 MoE、多节点自动容错或生产推理服务，应直接使用
Megatron-LM、DeepSpeed等成熟项目。

## 与Megatron Core对比

[Megatron-LM 做了哪些性能优化](docs/megatron-optimizations.md)

## 快速开始

需要 Python 3.11+、PyTorch 2.6+ 和
[uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/pyy233/Nano-Megatron.git
cd Nano-Megatron
uv sync --extra dev
```

先在 CPU 上运行一个无需数据文件的训练 smoke：

```bash
uv run nano-megatron-train \
  --config examples/configs/gpt_single_cpu.yaml \
  --max-steps 5
```

运行测试和静态检查：

```bash
uv run pytest -q
uv run ruff check .
```

## TinyStories：从 tokenizer 到 8 卡训练

### 1. 训练 tokenizer

下面两条脚本会下载固定版本的 TinyStories，确定性抽取 500,000 篇故事，并训练 8192
词表的 byte-level BPE：

```bash
uv run python scripts/download_tinystories.py

uv run python scripts/train_tinystories_tokenizer.py \
  --threads 16
```

检查 tokenizer 的 Unicode、special token 和 round-trip：

```bash
uv run nano-megatron-tokenizer inspect \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --text 'Once upon a time, 小猫 said hello 👋' \
  --add-eos
```

### 2. 构建完整 mmap 语料

该脚本流式生成完整 train/validation mmap，不会把全部 token stream 装入 Python 列表：

```bash
uv run python scripts/prepare_tinystories_corpus.py
```

输出位于：

```text
data/processed/tinystories-full-8k/
├── train.mmap/
└── validation.mmap/
```

### 3. 启动 8 卡训练

正式配置是 311M、TP2×PP2×DP2、global batch 256：

```bash
OMP_NUM_THREADS=1 \
uv run torchrun --standalone --nproc-per-node=8 \
  -m nano_megatron.cli.train \
  --config examples/configs/gpt_tinystories_l40s_8gpu.yaml
```

配置默认启用 W&B。先安装 tracking extra 并登录，或者在训练命令末尾禁用：

```bash
uv sync --extra tracking
uv run wandb login
```

```bash
--set wandb.enabled=false --set wandb.mode=disabled
```

特定 RunPod L40S 节点曾需要设置 `NCCL_P2P_DISABLE=1`；这不是通用要求。换机器后应先运行
`scripts/validate_nccl.py`。

### 4. 恢复训练

`--resume` 指向一个完整的 `step_*` 目录；`--max-steps` 是绝对停止 step：

```bash
CKPT=checkpoints/tinystories-l40s-311m-tp2-pp2-dp2/<timestamp>/step_00010000

OMP_NUM_THREADS=1 \
uv run torchrun --standalone --nproc-per-node=8 \
  -m nano_megatron.cli.train \
  --config examples/configs/gpt_tinystories_l40s_8gpu.yaml \
  --resume "$CKPT" \
  --max-steps 10488
```

恢复会创建新的时间戳 checkpoint 目录和新的 W&B run，并从框架自己的
`metrics.jsonl` 回放历史指标。

### 5. 导出并生成文本

把 TP/PP checkpoint 合并成单卡推理 artifact：

```bash
uv run nano-megatron-export \
  --checkpoint checkpoints/<run>/<timestamp>/step_XXXXXXXX \
  --output exported/tinystories-311m
```

在 CPU 或单张 GPU 上生成：

```bash
CUDA_VISIBLE_DEVICES=0 uv run nano-megatron-generate \
  --model exported/tinystories-311m \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt 'Once upon a time, there was a little fox' \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 1234
```
