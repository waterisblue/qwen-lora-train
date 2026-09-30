# Qwen3-8B LoRA 文本分类微调

基于 **昇腾 NPU（torch_npu）** + **Qwen3-8B** + **LoRA** 的中文短文本分类微调工程，包含完整的两段式流程：

1. **训练** —— 在 NPU 上用 LoRA 微调 Qwen3-8B，让模型学会"输入一段文本，只输出类别名称"。
2. **推理** —— 加载基座模型 + LoRA 适配器，通过 FastAPI 提供 HTTP 分类接口。

适用场景举例：银行流水摘要分类（货款 / 技术服务 / 广告宣传 / 其他）、110 接警文本分类（交通警情 / 网络诈骗 / 火灾警情 / 政务咨询）等**标签集合固定、输出需要极短**的分类任务。

---

## 目录结构

```
qwen-lora-train/
├── Dockerfile            # 基于 vllm-ascend:v0.9.1 的镜像，预装训练依赖
├── docker_start.txt      # docker run 启动命令（含 NPU 设备与驱动挂载）
├── requirements.txt      # Python 依赖（transformers / peft / datasets ...）
├── train.jsonl           # 示例数据集（银行流水分类）
├── README.md
└── src/
    ├── train.py          # LoRA 微调脚本
    ├── api.py            # FastAPI 推理服务
    ├── command.txt       # 环境自检脚本（验证 NPU 与模型能否正常加载）
    └── train.jsonl       # 示例数据集（110 接警分类）
```

运行时的目录约定（容器内路径）：

```
/workspace
├── models/Qwen3-8B/                  # 基座模型（需自行下载）
├── datasets/train.jsonl              # 训练集
├── datasets/val.jsonl                # 验证集
└── output/qwen-lora-classifier/      # LoRA 输出目录（自动创建）
```

`/workspace` 由宿主机目录挂载而来（见 `docker_start.txt` 中的 `-v /data/train_qwen:/workspace`），因此模型、数据、训练产物都持久化在宿主机上。

---

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | 昇腾 NPU（Atlas 800 系列等），已安装配套驱动与固件 |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.9.1`（已内置 torch / torch_npu） |
| 基座模型 | Qwen3-8B，需提前下载到 `/workspace/models/Qwen3-8B` |
| Python 依赖 | `transformers==4.51.3`、`peft==0.15.2`、`datasets==3.5.0`、`accelerate==1.6.0` |

> 训练脚本依赖镜像内自带的 `torch_npu`，因此 `requirements.txt` **刻意不包含 torch / torch-npu**，避免 pip 误升级导致与驱动版本不匹配。

---

## 快速开始

### 1. 构建镜像

```bash
docker build -t qwen-lora-train .
```

### 2. 启动容器

使用 `docker_start.txt` 中的命令（已配置 NPU 设备挂载、host 网络与 64G 共享内存）：

```bash
docker run -it \
  --name qwen-lora-train \
  --privileged \
  --network host \
  --security-opt label=disable \
  --shm-size=64g \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /data/train_qwen:/workspace \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /usr/local/openmpi:/usr/local/openmpi \
  quay.io/ascend/vllm-ascend:v0.9.1 \
  bash
```

### 3. 准备模型与数据

```bash
# 宿主机 /data/train_qwen/ 下准备：
#   models/Qwen3-8B/        <- 基座模型权重
#   datasets/train.jsonl    <- 训练集
#   datasets/val.jsonl      <- 验证集
```

### 4. 环境自检（可选但强烈建议）

进容器后先跑一遍 `src/command.txt`，确认 NPU 可见且模型能加载：

```bash
python src/command.txt
```

期望输出包含 `npu: True` 与 `MODEL LOAD OK`。

### 5. 启动训练

```bash
cd /workspace
python src/train.py
```

单卡训练。多卡（DDP）训练用：

```bash
torchrun --nproc_per_node=8 src/train.py
```

训练结束后 LoRA 适配器与 tokenizer 保存在 `/workspace/output/qwen-lora-classifier`。

### 6. 启动推理服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## 数据格式

JSONL，每行一个样本，仅需两个字段：

```json
{"text": "对手户名：某科技有限公司，摘要：支付系统开发费，备注：企业网银", "label": "技术服务"}
{"text": "对手户名：某广告有限公司，摘要：支付宝推广服务费，备注：企业网银", "label": "广告宣传"}
{"text": "对手户名：某贸易有限公司，摘要：支付货款，备注：企业网银", "label": "货款"}
{"text": "对手户名：日本乐日影视制作有限公司，摘要：支付平台，备注：企业网银", "label": "其他"}
```

- `text`：待分类文本，可长可短，超长会被截断到 `MAX_LENGTH`（默认 1024 token）。
- `label`：类别名称，**直接作为模型生成的目标字符串**，因此请保持简短（建议 ≤ 10 个字），且标签集合在训练集/验证集中保持一致。

`src/train.jsonl` 是一份 110 接警场景的示例数据，可作格式参考。

---

## 训练细节

### Prompt 模板

训练与推理使用**完全相同**的模板（这是分类效果稳定的关键）：

```
你是一个文本分类模型。
请根据输入文本判断其所属类别。
只输出类别名称，不要输出解释。

输入文本：
{text}

类别：
```

模型需要生成的答案就是 `label` 本身。

### Loss 掩码

`preprocess_function` 将 prompt 部分的 label 全部置为 `-100`，即**只在答案（类别名 + EOS）上计算 loss**：

```
input_ids = prompt_tokens + answer_tokens + [eos]
labels    = [-100] * len(prompt) + answer_tokens + [eos]
```

这样模型学的是"给定文本 → 输出类别"，而不是复述输入。

### LoRA 配置

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `r` | 16 | 秩 |
| `lora_alpha` | 32 | 缩放系数，`alpha/r = 2` |
| `lora_dropout` | 0.05 | |
| `bias` | `none` | |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 覆盖 Qwen 的注意力与 MLP 全部线性层 |

### 关键超参

| 参数 | 默认值 | 位置 |
| --- | --- | --- |
| 训练轮数 | 3 | `NUM_EPOCHS` |
| 学习率 | 2e-4（cosine，warmup 5%） | `LEARNING_RATE` |
| 单卡 batch size | 1 | `PER_DEVICE_TRAIN_BATCH_SIZE` |
| 梯度累积 | 8（等效 batch = 8 × 卡数） | `GRADIENT_ACCUMULATION_STEPS` |
| 精度 | `fp16`（NPU 上推荐，`bf16=False`） | `TrainingArguments` |
| 梯度检查点 | 开启 | 省显存，换来约 20% 速度损失 |
| 优化器 | `adamw_torch` | |
| 日志 / 评估 / 保存间隔 | 10 / 500 / 500 step | `LOGGING_STEPS` 等 |
| checkpoint 保留数 | 3 | `save_total_limit` |

以上常量都集中在 `src/train.py` 顶部，按需修改即可。

---

## 推理 API

服务启动后监听 `0.0.0.0:8000`，交互式文档在 `/docs`。

### `POST /classify`

请求：

```json
{"text": "有人在网上发布涉诈信息"}
```

响应：

```json
{"label": "网络诈骗"}
```

示例：

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "某小区发生火灾，请求消防处理"}'
```

### `GET /health`

```json
{"status": "ok", "npu": true}
```

> 推理使用贪心解码（`do_sample=False`），保证同一输入结果稳定可复现；`max_new_tokens=20` 足以覆盖短标签。

---

## 常见问题

**NPU 不可用 / `RuntimeError: NPU 不可用`**
容器启动参数不完整，或宿主机驱动未装好。先执行 `npu-smi info` 确认卡的状态，再核对 `docker_start.txt` 中的 `--device` 与驱动 `-v` 挂载是否齐全。

**`pip install` 后 torch 与驱动不匹配**
`requirements.txt` 已刻意不含 torch，如手动安装过，请恢复镜像内的版本（重建容器最省事）。

**LoRA 训练 loss 不下降 / 输出乱码**
优先检查三点：prompt 模板在训练和推理中是否完全一致；`label` 是否简短且候选集合固定；`fp16` 下是否出现溢出（可尝试关闭梯度检查点或调低学习率）。

**换了标签集合**
必须重新训练。LoRA 学到的是"这套标签 + 这套 prompt"，新增类别不会自动生效。

**多卡训练显存不足**
保持 `PER_DEVICE_TRAIN_BATCH_SIZE=1` 并增大 `GRADIENT_ACCUMULATION_STEPS`；或缩短 `MAX_LENGTH`（分类任务通常 256~512 已足够）。

---

## 二次开发提示

- **改为多标签或带解释输出**：修改 `build_prompt` 与 `label` 的内容即可，训练逻辑无需改动（loss 掩码自动适配）。
- **合并 LoRA 到基座权重**：用 `peft` 的 `merge_and_unload()`，可换来更快的推理速度，代价是失去适配器可插拔性。
- **服务化建议**：`api.py` 是单进程同步实现，生产环境应配合 `--workers` 或多实例 + 前置负载均衡；模型加载在模块导入时完成，注意启动预热。
