---
license: apache-2.0
language:
- zh
pipeline_tag: text-ranking
base_model: metasequoiaime/pinyin-ime-reranker-25M
tags:
- chinese
- pinyin
- input-method
- reranking
- on-device
---

# yanliu-25m-v1

砚流 v0.1.0 技术预览提供的 25M 中文候选排序模型。该版本当时默认使用固定公共原版 4M 权重；当前版本见[项目 README](../../../README.md#模型)。本模型根据有限前文给输入法已经召回的真实候选排序；不解码拼音、不扩大召回、不注入答案，也不读取个人学习数据。

模型为 8 层、8 heads、宽度 448、12,288 字词表、context64 的字符 Transformer，线性矩阵使用逐输出行对称 int8 量化。源文件为 `sentence-model.safetensors`；配套 `sentence-model.xml`、`.bin` 和 `.openvino.json` 是同一源文件导出的固定 FP32 IR，供显式 GPU/NPU 使用。OpenVINO 原生运行库和驱动单独准备，附件不包含这些库。

源权重 SHA-256：`26b00f6376fd9dbc7b834678175406f3c4260b84a5a455b6b2050122c573fa7e`。其他文件的大小、哈希、上游固定修订和派生链见 [provenance.json](provenance.json)。解压目录保持现有安装器需要的 `sentence-model.*` 命名；安装及设备配置见[项目 README](https://github.com/NEX-S/yanliu/tree/v0.1.0#readme)。

已从同版本 GitHub Release 下载 `yanliu-25m-v1.tar.gz` 并核对外部 SHA256SUMS 后，可从源码根目录解压并选择此模型进行 CPU 安装：

```sh
mkdir -p .cache
tar -xzf /path/to/yanliu-25m-v1.tar.gz -C .cache
(cd .cache/yanliu-25m-v1 && sha256sum --check SHA256SUMS)
YANLIU_MODEL_DIR="$PWD/.cache/yanliu-25m-v1" ./packaging/install-local.sh
```

GPU/NPU 使用相同模型目录，再显式选择设备与单独准备的原生运行库；完整步骤见[安装说明](https://github.com/NEX-S/yanliu/blob/v0.1.0/packaging/README.md)。安装器会同时保留本目录中的模型卡、许可、署名及来源记录。

## 来源与修改

起点为公共 [metasequoiaime/pinyin-ime-reranker-25M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M/tree/75295b373a7ef0d14e9bebd32c1625c9cfcfea49)，固定修订 `75295b373a7ef0d14e9bebd32c1625c9cfcfea49`，原文件 SHA-256 为 `0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469`。上游已在权重 metadata 中声明 Apache-2.0 许可及 Chinese C4 / LCCC 训练语料署名。

2026-10-19，砚流将 context128 裁为 context64，仅截取位置表前 64 行，其余张量原样保留；随后在 312 条公开合成语境上完成固定 3 轮 CPU 监督微调。源权重保留上游 `license`、`attribution` 及父源哈希，增加训练配方绑定、轮次和优化步骤。完整 Apache-2.0 许可证见 [LICENSE](LICENSE)，完整上游署名与本项目修改声明见 [NOTICE](NOTICE)。砚流本次权重微调贡献同以 Apache-2.0 发布；训练代码与自写数据遵循仓库 GPL-3.0-or-later，二者许可分别适用。

## 数据与选择限制

冻结的训练、开发数据和小型复现入口保留在源码的 [`data/training/yanliu-25m-v1`](https://github.com/NEX-S/yanliu/tree/v0.1.0/data/training/yanliu-25m-v1)。训练数据含 180 条新拼音案例及 132 条已知公开同音词语境，后 132 条的提升仅表示训练拟合。298 条能参与实际短名单损失，另 14 条因目标不在短名单而不参与损失。

训练脚本在 44 条新拼音开发案例上按分组等权 top1、平局取最早轮次，实际选择 epoch 0。实验另行预先固定了 94 条混合开发案例（44 条新拼音加 50 条已知词新语境）的外部选模，按相同准则比较 epoch 0–3，最终选择 epoch 3。混合开发的逐例 top1 从父模型的 `73/94` 到 `77/94`，组等权值从 `149/180` 到 `467/540`；它用于选择，不能估计独立泛化。

合成前文来自开发侧固定教师 deepseek-flash，目标由 pypinyin 0.55.0 自动注音，另有自写公开开发语境。未使用私人对话、真实应用上下文或系统学习；不发布原始网关响应。所有已观测开发、原先 32 条留出及既有 common/context/replay 结果均只能作开发或回归证据。自动注音、合成语境与少量人工诊断不代表自然输入分布，不能把上述分数称为自然用户准确率。

## 复现与运行范围

公开源码保留实际 312/44/94 数据、哈希、固定优化参数、两阶段选模和父源裁剪入口，可重新执行训练和行为选择。发布权重的历史训练 metadata 绑定当时的源码、binary 与绝对输出路径；新工作目录、构建、源码或 safetensors 序列化可能改变文件字节，不能保证逐字节重训生成此发布 SHA-256。每次新运行应读取它自己的选择记录，不能强制指定 epoch 3 作为“复现成功”。

CPU 使用 int8 源权重；显式 accelerator 构建可用同源 IR。源/IR 哈希由 `.openvino.json` 绑定，使用前可核对本包 `SHA256SUMS`。GPU 使用 FP32，NPU 使用设备默认精度，近似并列时排序可能不同。附件没有提供新的桌面键到屏幕延迟或自然用户质量测量；它仍受砚流有界候选召回和提升门槛约束。
