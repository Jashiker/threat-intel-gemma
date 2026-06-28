# Cybersecurity LLM — 多模型 LoRA/QLoRA 微调 + 安全实战工具集

基于 ModelScope [Cybersecurity-Dataset](https://www.modelscope.cn/datasets/hcnote/Cybersecurity-Dataset) 对多款开源大模型做 LoRA/QLoRA 微调，构建网络空间安全 AI 应用：**智能代码审计**、**威胁情报分析**、**论文研究辅助**。

## 解决的问题

- **安全工具告警误报多**：SAST 工具扫出数百条告警，LLM 二次研判降误报
- **威胁情报难结构化**：非结构化文本自动提取 TTPs、IOC、风险评级
- **大模型单卡微调难**：27B 模型 QLoRA 4-bit，48GB 单卡可跑
- **国内网络访问难**：纯 ModelScope 流程，无需 HuggingFace

## 项目结构

```
├── gemma4_cybersec_lora_modelscope_single_gpu.ipynb   # Gemma-4-E4B-it LoRA 微调
├── gemma4_emotion_lora_modelscope_single_gpu.ipynb    # 原版情绪分类参考
├── qwen3.5-9b_cybersec_lora.ipynb                     # Qwen3.5-9B LoRA 微调（推荐）
├── qwen3.6-27b_cybersec_qlora.ipynb                   # Qwen3.6-27B QLoRA 微调
├── threat-intel.py                                    # 威胁情报分析 CLI
├── code-audit.py                                      # 代码安全审计 CLI
├── cybersec-research.py                               # 论文研究辅助 CLI
├── demo.png                                           # 效果截图
└── requirements.txt                                   # Python 依赖
```

## 模型对比

| 模型 | 参数量 | 加载方式 | 显存 | 适用场景 |
|------|--------|----------|------|----------|
| **Qwen3.5-9B**（推荐）| 9B | BF16 | ~18GB | 单卡首选，零兼容问题 |
| Qwen3.6-27B | 27.8B | QLoRA 4-bit | ~20GB | 追求更强能力 |
| Gemma-4-E4B-it | ~8B | BF16 | ~16GB | 多模态基座 |

## 三大应用

### 1. 智能代码审计

```bash
# 审计单个文件
python code-audit.py -f app.py

# 批量审计目录
python code-audit.py -f src/ --batch -o report.md

# Semgrep 二次研判降误报
semgrep --config=auto app.py --json > results.json
python code-audit.py --semgrep results.json --src . -o triage.md
```

输出：CWE-ID、严重等级、漏洞描述、修复代码，覆盖 Python/C/Java/Go 等 15+ 语言。

### 2. 威胁情报分析

```bash
python threat-intel.py --ioc "185.220.101.34"          # IOC 分析
python threat-intel.py --cve "CVE-2024-3094"           # CVE 解读
python threat-intel.py -f apt_report.txt -o report.md  # APT 报告分析
```

输出：风险等级、MITRE ATT&CK TTPs 映射、IOC 提取、受影响系统、处置建议、关联威胁。

### 3. 论文研究辅助

```bash
python cybersec-research.py --paper paper.txt          # 论文解读
python cybersec-research.py --review papers/           # 文献综述
python cybersec-research.py --idea "IoT固件安全"        # 研究思路
python cybersec-research.py --explain "差分隐私"        # 概念解释
```

支持：论文解读、文献综述、研究思路生成、实验评估、概念解释、头脑风暴。

## 快速开始

```bash
pip install -r requirements.txt
```

在 Jupyter 中打开任一 notebook，按 Cell 顺序执行即可完成微调 + 内置应用面板。

## 核心特性

- **tokenizer 精准过滤**：源头过滤长样本防 OOM
- **多模型支持**：Gemma-4 / Qwen3.5-9B / Qwen3.6-27B 一键切换
- **LoRA / QLoRA 双模式**：BF16 全量 + 4-bit 量化
- **零额外显存应用**：复用已加载模型，代码审计/威胁情报/论文研究面板直接用
- **ROCm 兼容**：`adamw_torch` 优化器，AMD GPU 友好
- **纯 ModelScope**：模型和数据集均从魔搭下载

## 硬件要求

- GPU：单卡 ≥ 18GB（9B）/ ≥ 20GB（27B QLoRA）
- 支持 NVIDIA CUDA / AMD ROCm
- 默认 BF16，27B 自动启用 QLoRA 4-bit

## 模型与数据

- **基座模型**：Qwen3.5-9B / Qwen3.6-27B / Gemma-4-E4B-it
- **数据集**：[hcnote/Cybersecurity-Dataset](https://www.modelscope.cn/datasets/hcnote/Cybersecurity-Dataset)（全网最大开源网络安全数据集，~3.28GB）
- **微调方式**：LoRA（r=16，alpha=32），仅训练 ~0.6% 参数

## License

MIT
