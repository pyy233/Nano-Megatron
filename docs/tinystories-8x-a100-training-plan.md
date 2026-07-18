# TinyStories 311M / 8×A100 全量训练计划

## 1. 目标与边界

目标是在单机 8×A100 上，用已经训练好的 8192 byte-level BPE，从完整 TinyStories
训练集开始，完成一个 310,821,888 参数的 dense GPT 长训练，并且对数据、分布式语义、
checkpoint 恢复、吞吐、验证集 loss/perplexity 和生成质量逐项验收。

首个正式拓扑固定为：

```text
TP=2 × PP=2 × DP=2 = 8 GPUs
CP=1, EP=1
```

这条路径已经在 8×A100 上做过小模型 30-step BF16 correctness、validation、
checkpoint/load，以及 PP overlap/ZeRO 扩展验证。这里仍然要用最终 311M 模型和完整 mmap
重新做 smoke、恢复和长稳测试；已有短跑不能替代正式训练验收。

本计划不把 Transformer Engine、MoE、CP、ZeRO、VP/interleaved 或多节点加入首个长跑。
它们会同时增加变量，却不是让完整 TinyStories 首次收敛所必需的条件。

## 2. 当前起点与缺口

已经就绪：

- 完整 train TXT：`data/raw/tinystories/TinyStories-train.txt`；
- 2,119,489 篇非空 train stories；
- 500,000 篇确定性 tokenizer 样本；
- 严格 8192 词表 tokenizer：`data/tokenizers/tinystories-8k-500k/`；
- mmap corpus、无限多 epoch sampler、validation、cosine LR、W&B、`metrics.jsonl`、
  时间戳 checkpoint 和精确恢复实现。

正式训练前仍缺：

1. 固定 revision、大小和 SHA256 的官方 validation source；
2. 完整 train 和独立 validation 的 mmap artifact；
3. 面向 8×A100 的正式配置文件；
4. 用最终 GPU checkpoint 对已经实现的 export/generate CLI 做真实 A100 smoke；
5. 训练 target 不落在 save interval 时的 final checkpoint，以及最低 validation loss
   checkpoint 的保护机制。

500k JSONL 只用于训练 tokenizer，不能拿它冒充完整 GPT 训练语料。

## 3. 推荐基线规格

### 3.1 模型

| 字段 | 值 |
|---|---:|
| layers | 24 |
| hidden size | 1024 |
| FFN hidden size | 2736 |
| attention heads | 16 |
| KV heads | 16 |
| sequence length | 512 |
| vocabulary | 8192 |
| dropout | 0.0 |
| tied embedding/head | true |
| 精确参数量 | 310,821,888 |

模型使用仓库当前的 RMSNorm、RoPE、SwiGLU 和 PyTorch SDPA。当前后端没有
Transformer Engine 的线性层/Norm/MLP 融合，因此不能预先套用 Megatron-Core 的吞吐
数字，最终容量和速度必须在目标机器上实测。

### 3.2 并行与数值

```yaml
parallel:
  tensor: 2
  pipeline: 2
  context: 1
  expert: 1
  data: 2
  order: [tp, cp, ep, dp, pp]
  sequence_parallel: true

precision:
  params: bfloat16
  compute: bfloat16
  grad_reduce: float32

kernels:
  backend: torch

data_parallel:
  mode: ddp
  bucket_bytes: 268435456
  overlap_grad_reduce: true
  reshard_after_forward: true

pipeline:
  schedule: 1f1b
  overlap_p2p: true
  virtual_stages_per_rank: 1
  dynamic_activation_shapes: false

activation_checkpoint:
  mode: none
```

首跑用 DDP，而不是为了“更高级”切到 ZeRO。TP2×PP2 后每 rank 只持有约四分之一模型，
考虑 pipeline 两端各自持有 tied embedding/head，每 rank 实际约 7,980 万参数。311M 基线
完全不需要依赖 ZeRO 才能放入 A100；只有后续把模型显著放大时才重新比较 ZeRO-1，不要在
正式长跑前临时切换参数生命周期。

### 3.3 batch 与优化器

L40S 实测选定配置：

```yaml
training:
  micro_batch_size: 8
  gradient_accumulation_steps: 16
  seed: 1234
  log_interval: 10

optimizer:
  name: adamw
  lr: 0.0003
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  clip_grad_norm: 1.0

lr_scheduler:
  schedule: cosine
  min_lr: 0.00003
```

DP=2 时：

```text
global batch = 8 × 16 × 2 = 256 sequences
tokens / optimizer step = 256 × 512 = 131,072
```

`warmup_steps` 和 `decay_steps` 要在 train mmap 产出、知道真实 token count 后写死；不要
用估算值发布正式配置。初始训练 horizon 为 3 个 corpus epoch，warmup 取总步数的约 2%，
至少 100 steps，cosine decay 到初始 LR 的 10%。

### 3.4 数据、验证与 checkpoint

计划产物路径：

```text
data/processed/tinystories-full-8k/
├── train.mmap/
├── validation.mmap/
└── manifest.json
```

正式配置使用：

```yaml
data:
  path: null
  mmap_path: data/processed/tinystories-full-8k/train.mmap
  text_path: null
  tokenizer:
    path: data/tokenizers/tinystories-8k-500k
    append_eos: true
  num_workers: 2
  shuffle: true
  packed_sequences: false

validation:
  interval: 250
  batches: 32
  data:
    path: null
    mmap_path: data/processed/tinystories-full-8k/validation.mmap
    text_path: null
    tokenizer:
      path: data/tokenizers/tinystories-8k-500k
      append_eos: true
    num_workers: 2
    shuffle: false
    packed_sequences: false

checkpoint:
  directory: checkpoints/tinystories-a100-311m-tp2-pp2-dp2
  save_interval: 500
  async_save: false
  keep_last: 3

wandb:
  enabled: true
  project: nano-megatron-tinystories
  mode: online
```

`validation.batches=32` 需要在 validation mmap 生成后再次确认容量；通用计算是：

```text
validation_batches = min(32, floor(validation_samples / 256))
```

首次长稳测试先用同步 checkpoint，测出单次保存时长和实际磁盘大小。确认 host RAM、磁盘和
异步发布路径后，才考虑把 `async_save` 改为 `true`。建议开跑前至少预留 150 GB checkpoint
空间；首个 step checkpoint 生成后再以实测值修正预算。

W&B 用新 run 记录每次 CLI 启动；`wandb.name` 只用于人类识别。恢复会创建新时间戳目录和
新的 W&B 逻辑子 run，同时从原 checkpoint 所在目录的 `metrics.jsonl` 回放到恢复 step，
不会使用 W&B private-preview 的 `fork_from`。

## 4. 分阶段执行

每一阶段都是 gate；上一阶段未通过，不进入下一阶段。

### Gate A：机器与仓库预检

1. 确认是同一节点 8 张 A100，并记录 40 GB/80 GB、SXM/PCIe、驱动、CUDA、Torch、NCCL。
2. 用 `nvidia-smi topo -m` 确认 GPU 间拓扑；数据和 checkpoint 放本地 NVMe。
3. 检查 host RAM、磁盘空间、W&B 登录和系统时间。
4. 执行 `uv sync --frozen --extra tracking`、完整测试、`uv run ruff check .`、
   `uv lock --check` 和 `git diff --check`。
5. 默认不设置 `NCCL_P2P_DISABLE=1`。A100 环境已经验证过原生 NCCL P2P；只有真实复现
   collective 卡死并保存 NCCL 日志后，才把该变量作为诊断回退。

通过条件：8 卡均可见，拓扑符合租用规格，测试通过，空闲磁盘满足数据和至少三个 checkpoint。

### Gate B：构建完整 mmap

先补一个可复用的全量数据准备脚本，计划接口为：

```bash
uv run python scripts/prepare_tinystories_corpus.py \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --output-dir data/processed/tinystories-full-8k
```

这是待实现接口，不是仓库当前已有命令。脚本需要：

- 复用已校验的 train TXT，固定并下载官方 validation TXT；
- 按 `<|endoftext|>` 流式读取，不把所有故事装入列表；
- 直接或经原子临时 JSONL 流式编码为两个 mmap；
- 拒绝空故事、重复覆盖、tokenizer fingerprint 不匹配和 train/validation 复用同一 artifact；
- 输出 documents、tokens、tokens SHA256、logical fingerprint、tokenizer fingerprint 和
  source provenance 到 `manifest.json`；
- 成功自校验后再原子发布，并支持完整 artifact 的幂等复用。

通过条件：train/validation fingerprint 不同，vocab=8192，append_eos=true，token IDs 全部
合法，tokenizer fingerprint 精确一致，两个 artifact 均能被 inspect 和 DataLoader 打开。

得到 `T_train` 后计算：

```text
samples_per_epoch = floor((T_train - 1) / 512)
steps_per_epoch   = floor(samples_per_epoch / 256)
max_steps_3epoch = 3 × steps_per_epoch
warmup_steps      = max(100, ceil(0.02 × max_steps_3epoch))
decay_steps       = max_steps_3epoch
```

把这些整数写入计划中的
`examples/configs/gpt_tinystories_a100_8gpu.yaml`，再提交配置文件。

### Gate C：最终配置 8 卡 smoke 与恢复

统一启动形式：

```bash
OMP_NUM_THREADS=1 uv run torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m nano_megatron.cli.train \
  --config examples/configs/gpt_tinystories_a100_8gpu.yaml
```

分三次执行：

1. `--max-steps 1`：验证 311M 模型构建、显存和第一步 collective；
2. 跑到 step 10，并临时设置 `checkpoint.save_interval=10`、
   `validation.interval=5`、`validation.batches=4`；
3. 从打印出的 `step_00000010` 精确路径恢复到绝对 step 20。

恢复命令使用 `--resume <exact-step-path> --max-steps 20`。`--max-steps` 是绝对停止 step，
不是“再训练 20 步”。

通过条件：

- 8 rank 无 hang、OOM、NaN/Inf、shape mismatch 或 collective timeout；
- train/validation loss 有限，grad norm 有限；
- step 10 checkpoint 完整可加载；
- 恢复后第一步为 11，scheduler、consumed tokens、epoch/offset 和数据 fingerprint 连续；
- 新 `metrics.jsonl` 精确包含旧 run 到 step 10 的前缀，然后追加 step 11–20；
- 新 W&B run 能显示回放历史和续训指标。

### Gate D：固定 global batch 的吞吐 sweep

所有 case 保持 global batch=256、seq=512、模型、seed、数据、LR 和拓扑不变，只改：

| case | micro batch | accumulation | 理论 PP2 bubble |
|---|---:|---:|---:|
| A | 8 | 16 | 约 5.9% |
| B（原起点） | 16 | 8 | 约 11.1% |
| C | 32 | 4 | 约 20.0% |
| D（可选） | 64 | 2 | 约 33.3% |

每个 case 跑 80 steps：前 20 步 warmup，不保存 checkpoint、不做 validation、不启用 W&B，
比较后 60 步的 median tokens/s、p10 tokens/s、峰值显存和 GPU utilization。再对最快的 batch
case 比较 `num_workers=2/4/8`。基准期间在另一个终端记录 `nvidia-smi dmon -s pucm`；CLI
临时覆盖 `checkpoint.save_interval=0`、`validation.interval=0` 和 `wandb.enabled=false`。

8×L40S 实测结果（每组 80 step，统计 step 30–80 的六个 10-step 窗口）：

| case | median tokens/s | p10 tokens/s | peak allocated/reserved GiB |
|---|---:|---:|---:|
| A | 126,716.5 | 126,029.0 | 3.351 / 3.881 |
| B | 126,595.9 | 126,460.7 | 6.141 / 6.828 |
| C | 112,758.7 | 112,571.6 | 11.558 / 12.191 |
| D | 85,810.4 | 85,681.8 | 22.398 / 23.010 |

A 与 B 的 median 只差 0.095%，因此按预定规则选择 A：micro batch 8、accumulation 16。
在 A 上 `num_workers=2/4/8` 的 median 分别为 126,535.2 / 126,716.5 / 126,426.4
tokens/s，差异小于 0.3%；正式配置使用 2，减少整机 DataLoader 子进程数。

不要只凭理论 bubble 选 A。过小 micro batch 会降低 GEMM 效率、增加 Python/schedule 开销；
真正选择标准是目标机器的稳定 tokens/s。最终配置采用不 OOM、无明显显存碎片、且 median
tokens/s 最高的组合。若两组差距小于 3%，优先显存余量更大的组。

311M 的 baseline 在 A100 上不应接近显存上限。如果更大的 micro batch OOM，按以下顺序处理：

1. 降低 micro batch 并等比例提高 accumulation，保持 global batch=256；
2. 开 `activation_checkpoint.mode=selective`；
3. 再考虑 `mode=full`；
4. 不为了塞入 64×2 而引入 ZeRO；回到吞吐更好的较小 micro batch。

### Gate E：500-step 长稳与二次恢复

使用 Gate D 胜出的最终配置连续跑 500 steps；validation 每 100 steps、8–16 batches，
checkpoint 在 step 250 和 500 保存。随后从 step 500 恢复再跑至少 100 steps。

通过条件：

- 无 NCCL 卡死、CUDA OOM、非有限 loss/grad norm 或持续吞吐衰减；
- validation loss 相比首次评估明确下降；
- checkpoint 保存后的吞吐下降只出现在预期区间，随后恢复；
- step 500 恢复后数据、LR、metrics 和 loss 连续；
- 磁盘增长、保存耗时和 host RAM 峰值已经量化。

该阶段相当于最终配置的长时间系统验收。未通过时不要直接增加正式训练时长。

### Gate F：正式 3-epoch 训练

用 Gate B 计算出的 `max_steps_3epoch` 开始正式训练：

- log interval：10；
- validation interval：250，默认 32 batches；
- checkpoint interval：500，保留最近 3 个；
- W&B online，同时以每个时间戳 run 目录中的 `metrics.jsonl` 为框架权威历史；
- 保留 validation loss 最低的 checkpoint，不只保留最后一个 checkpoint。

当前 `keep_last` 只按时间保留最近 checkpoint，不会自动保留 best validation checkpoint。
正式开跑前应补“best checkpoint pin/copy manifest”能力，或者由外部监控在 validation 创新低时
记录并保护对应路径。

Trainer 当前也不会在任意 target step 自动补 final checkpoint；只有 step 能整除
`checkpoint.save_interval` 时才保存。正式开跑前应补“结束时若尚未保存则保存一次”的幂等
逻辑。若暂时不改代码，则每次计划停止点必须显式对齐保存 interval，但这只是临时约束。

3 epoch 后：

- 若 validation loss 仍稳定下降，按 1 epoch 为单位把绝对 `max_steps` 延长，最多先到 5 epoch；
- 若 validation loss 连续多次无改善或上升，停止扩展，选择最低 validation loss checkpoint；
- 不因 train loss 继续下降就认定模型仍在变好。

若从 3 epoch checkpoint 继续训练，原 cosine horizon 已结束，后续会在 `min_lr` 上继续；
不要修改 checkpoint 所绑定的旧 scheduler 历史来“回拉”学习率。若事先决定必须跑满 5 epoch，
应在 step 0 就把 5-epoch decay horizon 固定下来，而不是中途改变配方。

## 5. 最终质量验收

“训练跑完”和“模型训练好”不是同一件事。最终报告至少包含：

1. 最低 validation loss/perplexity、所在 step 和所消费 token 数；
2. 全程 train/validation 曲线、LR、grad norm、tokens/s 和 checkpoint pause；
3. 8 卡平均/分卡 GPU utilization、峰值显存、总训练时间和有效 tokens/s；
4. 从中途 checkpoint 恢复后的连续性证据；
5. 固定 prompt 集上的生成样例。

仓库已经提供 `nano-megatron-export`，可把 DDP/TP2×PP2×DP2/VP1 checkpoint 合并为
自包含单卡 artifact，并由 `nano-megatron-generate` 生成。正式宣称“训练好”前必须用最低
validation loss 的真实 GPU checkpoint 执行这条链路，并固定至少以下类型的 prompts：

- `Once upon a time, there was a little...`
- 多角色与对话；
- 因果/道德主题；
- validation 风格但未在 train 中完整出现的开头；
- Unicode 和换行输入。

生成验收记录 temperature、top-p、max new tokens 和 seed。至少检查语法连贯、主题遵循、
EOS 行为、明显循环，以及是否整段复制训练故事。

## 6. 成本与模型大小取舍

311M 是 TinyStories 数据规模与 TP2×PP2×DP2 工程目标之间的折中：它比 1.032B 更不容易
把训练资源用于记忆有限语料，同时又比约 120M 更适合保留三维 8 卡拓扑。它仍不会填满
A100-80GB 显存，计划追求的是正确收敛、稳定吞吐和可恢复性，而不是机械占满显存。

若目标后来变成“单位算力下最合适的 TinyStories 模型”，可以用同一 tokenizer、数据 split
和 token budget 补一个约 120M 的 DP8 对照；若目标变成纯分布式压力测试，再另建 1.032B
配置。两者都不属于本次正式基线。

## 7. 推荐执行顺序

```text
补全量 train/official-validation mmap 脚本
→ 生成并锁定 artifact manifest
→ 写入真实 token count 派生的正式 YAML
→ 1-step 311M allocation smoke
→ 10→20 step checkpoint/resume smoke
→ 固定 global batch 吞吐 sweep
→ 500→600 step 长稳/恢复
→ 正式 3 epoch
→ 按 validation 决定是否延长到最多 5 epoch
→ best checkpoint + 固定 prompts 生成验收
→ 可选 120M / DP8 数据效率对照组
```
