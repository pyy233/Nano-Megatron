# TinyStories 500k / 8K BPE 数据流水线

这条流水线用于从空工作区重建正式 tokenizer 输入。`data/` 被 Git 忽略；目录不存在时，
两个脚本会自行创建所需父目录。

## 1. 下载与精确抽样

```bash
uv run python scripts/download_tinystories.py
```

脚本按顺序尝试固定 revision 的 ModelScope 和 Hugging Face 地址。最终源文件必须同时满足：

| 字段 | 固定值 |
|---|---|
| 文件 | `TinyStories-train.txt` |
| 大小 | `1,924,281,556` bytes |
| SHA256 | `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f` |
| delimiter-defined records | `2,119,719` |
| 排除的空白 records | `230` |
| 可训练非空故事 | `2,119,489` |
| 分隔符 | `<\|endoftext\|>` |

下载先写入 `data/raw/tinystories/TinyStories-train.txt.part`。网络中断后再次执行会发送
HTTP Range 请求继续下载；如果镜像忽略 Range 并返回完整文件，只会重建脚本自己的
`.part` 文件。只有大小和 SHA256 都正确时，才原子发布为 `TinyStories-train.txt`。

原始 row 统计包含 230 个纯空白片段；它们不能进入框架的非空训练文本接口，所以脚本会
在 metadata 中记录并排除。抽样不是取前 50 万篇。脚本对非空故事的
`(seed=1234, source_index)` 计算版本化 BLAKE2b score，
选择 score 最小的严格 500,000 个 index。第一遍只保留 index/score，第二遍按原始顺序
写 JSONL，因此不会把 50 万篇正文同时放进内存。

输出：

```text
data/raw/tinystories/
├── TinyStories-train.txt
├── train_500k.jsonl
└── train_500k.metadata.json
```

metadata 记录源文件 fingerprint、抽样算法、seed、source/sample 文档数，以及 JSONL
大小和 SHA256。完整结果可重复执行并复用；只存在 JSONL 或只存在 metadata 时会拒绝
猜测性覆盖，需要人工检查后同时移除这两个文件再重建。

## 2. 训练严格 8192 词表

```bash
uv run python scripts/train_tinystories_tokenizer.py \
  --threads 16
```

默认训练设置：

```text
documents       = 500000
vocab_size      = 8192
min_frequency   = 2
tokenizer       = byte-level BPE
special tokens  = UNK/BOS/EOS/PAD
```

训练先写入同级临时目录。框架会重新加载 `tokenizer.json` 和 `metadata.json`，要求：

- trainer 实际消费 500,000 篇；
- artifact fingerprint 与 metadata 一致；
- 实际 `vocab_size` 恰好为 8192，而不是只把 8192 当作上限；
- sample JSONL 的 SHA256 与下载脚本 metadata 一致。

任一条件不满足都会删除临时目录，不发布目标 artifact。成功结果为：

```text
data/tokenizers/tinystories-8k-500k/
├── tokenizer.json
├── metadata.json
└── training.json
```

`training.json` 绑定输入 JSONL 的绝对路径、SHA256、文档数、目标词表、min frequency 和
tokenizer fingerprint。重复执行相同命令会验证并复用；若目标目录来自不同语料或配置，
脚本拒绝覆盖。

## 3. 验收与训练配置

编码/解码检查：

```bash
uv run nano-megatron-tokenizer inspect \
  --tokenizer data/tokenizers/tinystories-8k-500k \
  --text 'Once upon a time, 小猫 found a blue key 👋' \
  --add-bos \
  --add-eos
```

GPT 配置必须精确绑定：

```yaml
model:
  vocab_size: 8192

data:
  tokenizer:
    path: data/tokenizers/tinystories-8k-500k
    append_eos: true
```

500k JSONL 只是 tokenizer 训练样本，不是完整 GPT 训练 token corpus。正式 GPT 训练仍应
把完整训练/验证文本分别预处理成 mmap artifact。

## 4. 本机资源预期

在 i9-13900H / 30 GiB RAM 上，建议关闭占用大量内存的应用。完整源文件约 1.9GB，
500k JSONL 通常约 400～500MB；Rust tokenizer 会使用多线程和数 GB 到十几 GB 内存。
`--threads 16` 是当前机器的推荐起点。如果发生明显 swap 或温度降频，可改为
`--threads 12`；线程数只影响性能，不改变输入、验收规则或目标词表大小。
