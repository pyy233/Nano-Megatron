# TinyStories 311M：8×L40S 实测与正式训练手册

这份手册只针对 2026-07-18 验证的 RunPod：8 张 NVIDIA L40S，每卡可见 46,068MiB，
双 NUMA PCIe、无 NVLink。正式配置是
[`examples/configs/gpt_tinystories_l40s_8gpu.yaml`](../examples/configs/gpt_tinystories_l40s_8gpu.yaml)：

- 310,821,888 参数，24 layers，hidden 1024，FFN 2736，16 heads；
- BF16，seq 512，TP2×PP2×DP2，sequence parallel，1F1B + P2P overlap；
- micro batch 8，gradient accumulation 16，global batch 256；
- 10,488 optimizer steps，即完整 train mmap 的 3 个 epoch；
- cosine LR 3e-4→3e-5，warmup 210 steps；
- validation 每 250 steps、32 global batches；checkpoint 每 500 steps。

## 1. 这台机器的强制通信设置

这台 Pod 的默认 NCCL P2P transport 会在连续 collective 中卡住：8 个进程不返回，
8 张卡保持 100% SM utilization，但不产生训练进展。`NCCL_P2P_LEVEL=PIX` 也会复现。
所有 8 卡命令必须设置：

```bash
NCCL_P2P_DISABLE=1
```

禁用后已通过 100 次 32MiB all-reduce、100 次 all-gather、100 次 ring P2P、500 次
barrier，以及下面的完整训练测试。这是当前 Pod 的实测约束，不表示所有 L40S 节点都应
禁用 P2P；换机器后应重新运行 `scripts/validate_nccl.py`。

## 2. 数据与 tokenizer

当前 RunPod 已有完整 artifact：

```text
data/processed/tinystories-full-8k/
├── .complete
├── manifest.json
├── train.mmap/
└── validation.mmap/
```

校验后的规模：

| split | stories | tokens | seq512 samples |
|---|---:|---:|---:|
| train | 2,119,489 | 458,322,556 | 895,161 |
| validation | 21,990 | 4,610,408 | 9,004 |

Tokenizer 是 500,000 篇确定性样本训练的 byte-level BPE，vocab 8192，fingerprint 为
`dac8b8de30b96fe3076c3af55ea501da06810ec175ede5dd47ca39bb4be373a1`。
已发布 artifact 存在时，`scripts/prepare_tinystories_corpus.py` 会先完整校验 mmap、manifest
和 tokenizer，再直接返回 `reused=true`；即使 1.924GB raw train TXT 已被清理，也不会为
复用操作重新下载它。

## 3. Batch sweep 与稳定性实测

四组测试固定模型、TP/PP/DP、数据顺序、seed、LR、seq512 和 global batch 256。每组
运行 80 step，统计 step 30–80 的六个 10-step 窗口：

| micro × accumulation | median tokens/s | p10 tokens/s | peak allocated/reserved GiB |
|---|---:|---:|---:|
| 8 × 16 | 126,716.5 | 126,029.0 | 3.351 / 3.881 |
| 16 × 8 | 126,595.9 | 126,460.7 | 6.141 / 6.828 |
| 32 × 4 | 112,758.7 | 112,571.6 | 11.558 / 12.191 |
| 64 × 2 | 85,810.4 | 85,681.8 | 22.398 / 23.010 |

8×16 与 16×8 的 median 只差 0.095%，因此选择显存更低、pipeline bubble 更小的 8×16。
`num_workers=2/4/8` 的 median 分别为 126,535.2 / 126,716.5 / 126,426.4 tokens/s，
差异小于 0.3%；正式配置用 2，减少整机 DataLoader 子进程数。

胜出配置已连续运行 500 step，并从 checkpoint 恢复到 step600：

| 指标 | 实测值 |
|---|---:|
| step>20 median tokens/s | 126,178.2 |
| 吞吐范围 | 122,432.4–126,919.5 |
| WORLD peak allocated/reserved | 3.352 / 3.883GiB |
| validation loss@100 | 4.5104 |
| validation loss@500 | 1.9859 |
| validation loss@600 | 1.8685 |
| validation perplexity@600 | 6.4787 |

500-step 过程中无 OOM、NCCL hang、NaN/Inf 或显存持续增长。step250 的同步 checkpoint
只让紧随其后的一个日志窗口降到约 122.4k tokens/s，之后恢复。step500→600 恢复后，
历史 `metrics.jsonl` 前缀逐对象一致，数据 offset、LR scheduler 和 token 计数连续。

按约 126k tokens/s 估算，10,488 steps 的纯训练计算约 3.0 小时；计入 validation、同步
checkpoint 和节点波动，建议为正式 3-epoch 运行预留 3.3–4 小时。短跑的 3.35GiB 是框架
记录的 CUDA tensor 峰值；`nvidia-smi` 还会显示 context/allocator 等额外占用。

## 4. 正式启动

先登录 W&B；测试期间远程没有保存本地机器的凭据：

```bash
cd /workspace/nano-megatron
.venv/bin/wandb login
```

然后从 step0 启动：

```bash
cd /workspace/nano-megatron

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m nano_megatron.cli.train \
  --config examples/configs/gpt_tinystories_l40s_8gpu.yaml
```

rank0 会打印实际的时间戳目录，例如：

```text
checkpoint run directory: checkpoints/tinystories-l40s-311m-tp2-pp2-dp2/20260718-...
```

该目录同时保存 `metrics.jsonl`、`best_validation.json` 和 `step_*`。`keep_last=1` 会保留
最新 checkpoint；若最低 validation loss 不在最新 step，还会额外保护 best checkpoint。

若暂时不使用 W&B，可在启动命令末尾加：

```bash
--set wandb.enabled=false --set wandb.mode=disabled
```

## 5. 监控与恢复

另开终端监控 GPU、磁盘和框架指标：

```bash
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader'
```

```bash
watch -n 10 'df -h /workspace'
```

```bash
tail -f checkpoints/tinystories-l40s-311m-tp2-pp2-dp2/<timestamp>/metrics.jsonl
```

异常中断后，选择存在 `.complete` 的最新精确 step 目录。`--max-steps` 是绝对停止 step，
不是“再跑多少步”：

```bash
RUN=checkpoints/tinystories-l40s-311m-tp2-pp2-dp2/<source-timestamp>
CKPT=$(find "$RUN" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort | tail -n 1)

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m nano_megatron.cli.train \
  --config examples/configs/gpt_tinystories_l40s_8gpu.yaml \
  --resume "$CKPT" \
  --max-steps 10488
```

恢复会创建新的时间戳 checkpoint run 和新的 W&B run，并从框架自己的 `metrics.jsonl`
回放历史指标。这是“逻辑 fork”，不依赖 W&B private-preview 的 `fork_from`。

`/workspace` 只有 20GB。正式运行前应确保至少约 6GB 可用；若会多次恢复，建议留 10GB
以上，或把已经验收且不再需要的旧测试 checkpoint 转存到持久卷。一个 311M TP2×PP2
checkpoint 实测约 1.8GB，单卡 BF16 导出约 594MB。

## 6. 导出与单卡推理

训练结束后可导出最新 checkpoint；质量验收通常还应按 `best_validation.json` 选择最低
validation loss 对应的 step：

```bash
RUN=checkpoints/tinystories-l40s-311m-tp2-pp2-dp2/<timestamp>
CKPT=$(find "$RUN" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort | tail -n 1)

.venv/bin/nano-megatron-export \
  --checkpoint "$CKPT" \
  --output exported/tinystories-l40s-311m
```

单卡生成：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/nano-megatron-generate \
  --model exported/tinystories-l40s-311m \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt 'Once upon a time, there was a little fox' \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 1234
```

step600 的真实导出已经在单张 L40S 上通过 5 类 prompts；能够生成连贯的角色、对话和
简单因果结构，Unicode prompt 正常，其中一组输出以 EOS 停止。600 step 只是正式训练前
质量 smoke，不能替代 3 个 epoch 后的 held-out validation 与固定 prompts 验收。
