# LoRA 微调 Gemma-4-E4B-it 网络安全数据集

基于 ModelScope 的 [Cybersecurity-Dataset](https://www.modelscope.cn/datasets/hcnote/Cybersecurity-Dataset) 对 Google Gemma-4-E4B-it 做 LoRA 微调，构建网络安全 AI 应用。

## 项目结构

```
├── gemma4_cybersec_lora_modelscope_single_gpu.ipynb  # LoRA 微调 Notebook
├── threat-intel.py                                    # 威胁情报分析 CLI 工具
├── demo.png                                           # 效果截图
└── requirements.txt                                   # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行微调 Notebook

在 Jupyter 中打开 `gemma4_cybersec_lora_modelscope_single_gpu.ipynb`，按 Cell 顺序执行：

- 从 ModelScope 下载 Gemma-4-E4B-it 模型和 Cybersecurity-Dataset 数据集
- LoRA 微调（单卡，BF16，~8B 参数）
- 微调前后效果对比

### 3. 使用威胁情报分析面板

```bash
# 交互面板
python threat-intel.py

# 快速 IOC 分析
python threat-intel.py --ioc "185.220.101.34"

# CVE 漏洞解读
python threat-intel.py --cve "CVE-2024-3094"

# 文件分析并导出报告
python threat-intel.py -f apt_report.txt -o analysis.md
```

## 功能

### 威胁情报分析面板

支持三类输入，输出六维结构化研判结果：

- **IOC 分析**：IP / 域名 / URL / 文件哈希
- **CVE 解读**：漏洞影响评估与缓解建议
- **自由情报分析**：APT 报告、安全日志、告警文本

输出维度：风险等级 | MITRE ATT&CK TTPs 映射 | IOC 提取 | 受影响系统 | 处置建议 | 关联威胁

## 硬件要求

- GPU：单卡 ≥ 16GB 显存（推荐 24GB+）
- 支持 NVIDIA CUDA / AMD ROCm
- 默认使用 BF16，AMD ROCm 下不使用 bitsandbytes

## 模型与数据

- **基座模型**：[google/gemma-4-E4B-it](https://www.modelscope.cn/models/google/gemma-4-E4B-it)（Gemma 4，instruction-tuned，~8B 参数）
- **数据集**：[hcnote/Cybersecurity-Dataset](https://www.modelscope.cn/datasets/hcnote/Cybersecurity-Dataset)（全网最大开源网络安全数据集，~3.28GB）
- **微调方式**：LoRA（r=16，alpha=32），仅训练 0.63% 参数

## License

MIT
