# Alignment Stress Lab（对齐鲁棒性实验室）

[English](README.md) | 简体中文

Alignment Stress Lab 用于研究奖励驱动的微调如何改变语言模型的安全行为与通用能力。
首批目标是 Qwen3.5-9B 和 Qwen3.8-27B 系列中已对齐的 Dense 检查点。
项目仍在开发中，目前没有经过验证的基准成绩。

## 项目目标

- 9B 与 27B 共用一套训练代码，用配置表达模型差异。
- 在本地只加载一份基础模型，通过启用或禁用 adapter 切换 Policy 与冻结的 Reference。
- 通过外部 Judge API 评估，无须在 DGX 上再加载一份 Judge 模型。
- Judge 请求失败时保留 `unknown`；无效生成按失败计入攻击成功率（ASR）的分母。
- 记录评判协议版本、缓存来源、Judge 覆盖率、安全与能力指标，便于复核。

以上是设计目标。目前已完成评估数据契约、初版 Judge 客户端和模型无关的 GRP-Oblit 损失，
并使用 CPU 合成张量测试。无效生成只识别空回答；更完整的生成质量检测、模型加载、
训练流程和基准集集成尚未完成。

## 研究边界

实验应在隔离、获授权的环境中进行。项目不会自动上传数据集、回答、adapter 或检查点。
盲测数据必须与训练及调参数据隔离。训练后的检查点不用于公开推理部署。

## 开发环境

使用 Python 3.11 或 3.12。Windows 上先选定已安装的解释器。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
```

依赖版本统一在 `pyproject.toml` 中维护。`requirements.txt` 安装基础包，
`requirements-dev.txt` 增加测试工具和用于合成数学测试的 CPU PyTorch，
`requirements-train.txt` 增加模型训练依赖。本地测试不使用模型权重。
在 DGX Spark 上，训练依赖应安装到经过验证的 NGC 衍生容器中，保留与 ARM64 兼容的
CUDA 与 PyTorch 环境。

数学约定见 [docs/grp-oblit-math.md](docs/grp-oblit-math.md)。

## Judge 配置

参考 [configs/judge.example.yaml](configs/judge.example.yaml)，在环境中配置
`JUDGE_MODEL`、`JUDGE_BASE_URL` 和 `JUDGE_API_KEY`。Judge 会将提示词与回答发送给
指定的外部 API。使用前需确认服务商允许处理对应研究数据。离线测试注入假客户端，不发送请求。

## 实验结果

目前不发布 ASR 或能力保留率声明。确定模型检查点、盲测基准、采样设置、Judge 协议与
分析规则并保留记录后，再加入可复核结果。

公开的[中文安全开发提示集](data/prompts/safety_baseline_dev_zh.txt)为合成数据，
每行一条，仅用于打通基线流程；它不是盲测基准，也不作为训练数据。

## 致谢

本项目借鉴了 [Abliterix](https://github.com/wuwangzhang1216/abliterix) 的 Judge、
缓存溯源和评估失败处理经验，以及 [Heretic](https://github.com/p-e-w/heretic) 的
adapter 与模型加载工程经验。[GRP-Obliteration 论文](https://arxiv.org/abs/2602.06258)
提出了研究问题；[社区 GRP 仓库](https://github.com/GuyNachshon/GRP-Obliteration)
提供了可检查的实现参考。本仓库独立实现；致谢不代表原作者认可本项目，也不代表其论文结果已在此复现。

## 许可证

本仓库采用 [Apache-2.0](LICENSE)。致谢项目仍受各自许可证约束；本仓库目前未包含其源代码。
