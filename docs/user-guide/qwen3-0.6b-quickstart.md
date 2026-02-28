# Qwen3-0.6B 快速上手（Pretrain + 基础评测）

本文给一条最短可跑通路径：

1. 选一个预训练数据集（本文用 `Salesforce/wikitext` 的 `wikitext-103-raw-v1`）
2. 把原始文本转成 Megatron 训练格式（`*_text_document.bin/idx`）
3. 用 `pretrain_gpt.py` 从零训练 Qwen3-0.6B
4. 做基础评测（验证集 loss/PPL）

## 0. 环境准备

在仓库根目录执行：

```bash
cd /Users/huangjc/code/Megatron-LM

# Megatron-LM 依赖
pip install --no-build-isolation .[mlm,dev]

# 下载 HF 数据集用
pip install datasets
```

## 1. 选择并导出数据集（WikiText-103）

这里把 HF 数据集导出成 `jsonl`，每行一个 `{"text": ...}`。

```bash
python3 - <<'PY'
from datasets import load_dataset
import json

out_path = "/tmp/wikitext103_qwen3.jsonl"
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

with open(out_path, "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        for x in ds[split]:
            t = x["text"].strip()
            if t:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print(f"saved: {out_path}")
PY
```

## 2. 预处理成 Megatron 数据格式

```bash
mkdir -p /tmp/qwen3_data

python3 tools/preprocess_data.py \
  --input /tmp/wikitext103_qwen3.jsonl \
  --output-prefix /tmp/qwen3_data/wikitext103_qwen3 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-0.6B \
  --workers 8 \
  --append-eod
```

会生成：

- `/tmp/qwen3_data/wikitext103_qwen3_text_document.bin`
- `/tmp/qwen3_data/wikitext103_qwen3_text_document.idx`

训练时 `--data-path` 用前缀：

```bash
export DATA_PATH=/tmp/qwen3_data/wikitext103_qwen3_text_document
```

## 3. 启动 Qwen3-0.6B Pretrain

下面命令是单机单卡的快速上手配置。模型结构参数来自仓库 `examples/post_training/modelopt/conf/Qwen/Qwen3-0.6B.sh`。

```bash
cd /Users/huangjc/code/Megatron-LM

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CHECKPOINT_DIR=/tmp/qwen3_0p6b_pretrain
mkdir -p ${CHECKPOINT_DIR}

# 1 卡快速验证版本（先跑通流程）
torchrun --nproc_per_node=1 pretrain_gpt.py \
  --use-mcore-models \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --num-layers 28 \
  --hidden-size 1024 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 16 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 128 \
  --seq-length 2048 \
  --max-position-embeddings 40960 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --normalization RMSNorm \
  --swiglu \
  --qk-layernorm \
  --disable-bias-linear \
  --no-masked-softmax-fusion \
  --no-rope-fusion \
  --no-bias-swiglu-fusion \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-0.6B \
  --make-vocab-size-divisible-by 1187 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --train-samples 200000 \
  --lr-decay-samples 200000 \
  --lr-warmup-samples 2000 \
  --lr 3.0e-4 \
  --min-lr 3.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --init-method-std 0.010 \
  --bf16 \
  --attention-backend auto \
  --data-path ${DATA_PATH} \
  --split 98,2,0 \
  --log-interval 10 \
  --eval-iters 20 \
  --eval-interval 200 \
  --save-interval 500 \
  --save ${CHECKPOINT_DIR} \
  --load ${CHECKPOINT_DIR}
```

## 4. 基础 evaluation

### 4.1 训练中验证（推荐）

上面的配置会每 `200` step 在验证集打印：

- `validation ... loss`
- `validation ... PPL`

这就是最基础的 pretrain 评测信号。

### 4.2 Eval-only（不继续训练）

训练后可单独跑一次验证：

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
  --use-mcore-models \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --num-layers 28 \
  --hidden-size 1024 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 16 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 128 \
  --seq-length 2048 \
  --max-position-embeddings 40960 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --normalization RMSNorm \
  --swiglu \
  --qk-layernorm \
  --disable-bias-linear \
  --no-masked-softmax-fusion \
  --no-rope-fusion \
  --no-bias-swiglu-fusion \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-0.6B \
  --make-vocab-size-divisible-by 1187 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --skip-train \
  --eval-iters 100 \
  --data-path ${DATA_PATH} \
  --split 98,2,0 \
  --load ${CHECKPOINT_DIR}
```

## 5. 常见坑

- `--data-path` 要指向前缀 `..._text_document`，不是 `.bin` 文件本身。
- 单卡先用 `--seq-length 2048`，显存不够可降到 `1024`。
- `WikiText-103` 只适合流程验证，不适合真正训练出强泛化模型。
- 如果是消费级 GPU 不支持 BF16，可把 `--bf16` 改为 `--fp16`。
