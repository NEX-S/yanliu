# 模型版本与来源复核

复核日期：2026-10-23。这里补充历史发布记录中缺失的来源条件，并区分公开默认权重、历史修复阶段和当前本机选定权重。固定发布包和 provenance 保留原字节，新增说明不替来源授予权利。

## 默认下载与当前本机权重

`scripts/fetch_assets.py` 按 `data/manifest.json` 获取 `models-v2` 发布的 4M 权重，SHA-256 为：

```text
625dfc10eeb0fd4a723765589e5ebe1a302f25a61ea923cb4b95265efac70293
```

历史本机修复阶段的权重 SHA-256 为：

```text
7904a8a42974b1f003a86c1a8422d6b8a4d5ef1237207cd0e1f627683886c638
```

当前本机选定并安装的是从该历史修复权重继续训练的版本，SHA-256 为：

```text
ca1f0279a72213510471b0f9bda7876b8dc6b40cdb2a4c4d7c227e737feebe05
```

当前模型仍是 4,478,592 参数、6 层、隐藏维度 192、4 个注意力头、字表 9,358、上下文 64 的字符模型，INT8 文件为 4,734,576 bytes。二维权重除位置嵌入外逐行 INT8 量化，行缩放和其余张量保留 FP32。当前模型包在本机 `results/broad-model-runtime-20261023/model-package/`，包内权重的实际哈希与 provenance 记录一致；它和历史修复权重均未作为新的公共模型版本发布，没有独立公开下载地址，默认下载不能取得它们。

本轮继续训练在 Arc B390 GPU 上使用 FP32 完成 3 轮、2,577 次更新，按固定 923 条已观测开发数据的分组首选指标选择第 3 轮。7,384 条训练输入由 4,096 条新增公开语料监督案例和去重后的既有案例组成。新增案例复用公开语料原文，使用固定的 pypinyin 0.55.0 和[十短语补充](../training/pinyin-phrases.README.md)生成注音；自动注音不等于人工金标。

训练使用源模型真实 CLI 菜单，包括有界神经召回实际进入评分的候选。源模型已正确首选的排序组使用 hinge 保持损失，其余排序组使用真实菜单上的排序交叉熵。实际评分候选中缺失、但字表可表示且上下文窗口容得下的目标，只通过目标后缀的字符负对数似然监督；这个训练目标不作为候选注入 CLI。普通文本项使用冻结父 4M 模型到当前模型的全字表 KL 保持，温度和系数均为 1，没有另加普通文本交叉熵或 25M 候选蒸馏。权重 header 中的 `training_precision` 仍继承为 `bf16`；本轮配方、runtime 和 provenance 均记录 `fp32`，实际训练精度以本轮记录为准，冻结模型包未改写。完整配方为本机 `results/broad-menu-training-20261023/experiment/control/training-run/recipe.json`，SHA-256 为 `4900581a4d0eac5f23a9bef73ad29d3fbf176dd24b52526a18d82cfaac7926c1`；选模和安装决定分别见同目录 `selection.json` 与 `results/broad-model-runtime-20261023/result.json`。

本次本机安装所用的 CPU release CLI SHA-256 为 `cb748660bd18546224df8d72235213247ecbd08021d5277e075c79ab9b68fcd6`，已安装 IBus binary SHA-256 为 `d2a3e35e488b56848e079dc769e02b1cc9b2a17d5948cc3dd5e18315c919680f`。本机 `results/scalar-logprob-20261023/result.json` 与 `system-verification.json` 记录这次 CPU 概率计算内核更新保留了 `ca1f0279…` 权重，既有回归的候选与评分报告一致；这不构成新的质量提升或独立泛化成绩。

CPU 首次安装未指定模型目录时读取 `data/models`；CPU 升级默认保留已安装模型及其来源记录。显式指定 `YANLIU_MODEL_DIR` 可更换模型，包括切换到仓库当前默认模型；GPU/NPU 安装与升级均要求显式指定源模型及匹配 IR/runtime。步骤见[安装说明](../../packaging/README.md)。

## 历史修复阶段的 321 条记录

在 `7904a8a4…` 修复阶段，用同一份当时的 release CLI 检查冻结的 321 条已确认问题，公开默认 `625dfc10…` 权重首选命中 62 条、前五命中 212 条、召回 319 条；历史修复 `7904a8a4…` 权重三项均为 321 条。输入文件 SHA-256 为 `831da28fa87e9b7eba8093f8b0c7fbb9b706166af6fe1639a21f7ae331fc150d`，该次 CLI SHA-256 为 `20dac1378202df53ec500cf3233d9b20f7ef3a1da171632e786fe933cab34132`。

这段比较仅是该阶段、上述输入和 binary 的记录，不是用当前 `ca1f0279…` 权重和最新 CLI 重新测得的结果。这 321 条参与过修复训练，结果用于确认已知问题是否修好，不能作为独立泛化成绩，也不能替代 README 中注明条件的历史留出评测。该阶段原始报告保存在本机 `results/release-readiness-20261023/` 与 `results/prefix-reuse-20261023/`，尚未整理为公开评测包。当前继续训练的训练集、已观测开发集和回归集也不能作为独立泛化证明。

## NaturalConv 来源更正

历史模型的 `LICENSE`、`NOTICE` 与冻结配方写有“数据卡未声明许可”。同一训练来源版本实际包含独立的许可证文件：

- 来源版本：`xywang1/NaturalConv@adb38574e8014ada1d8404247167582e76b22d05`。
- [固定版本 LICENSE](https://huggingface.co/datasets/xywang1/NaturalConv/blob/adb38574e8014ada1d8404247167582e76b22d05/LICENSE)：3,634 bytes，SHA-256 为 `b51724699c7602950328471609ef586995eb7ecb870a93bc45eea79cd1a9946e`。
- [作者仓库](https://github.com/naturalconv/NaturalConvDataSet)也明确说明非商业用途限制。

条款第 3 节只向个人、非商业科研、开发、测试及指定学术发表和技术活动展示等用途授予有限许可。第 4 节限制数据及其衍生作品的分发、转售和其他使用，并限制通过腾讯以外的来源提供数据。

这补齐了对实际条款的记录；训练权重是否属于条款中的衍生作品仍需单独确认。现有材料不足以宣称整份混合来源模型可无条件商用或自由再分发。本机修复版继承了父模型的训练来源，继续微调没有消除这些来源条件。

## 尚未齐备的发布材料

原配方还保留 COIG-CQIA 无统一许可声明、TNEWS `unknown` 的记录；Wikipedia 的文章归属和各来源原始声明需要随适用条件保留。CPED 的仓库许可与其电视对白来源分别记录。新增 DeepSeek Flash 合成样本也需要保留生成记录与适用的服务输出条件。

本机修复的冻结输入、训练配方、各阶段源码和验证报告目前分布于忽略目录 `results/`、`data/cache/`。已有文件和哈希能核验本机过程，干净 checkout 还不能取得整条修复链。整理新发布包时，应给出可取得的固定输入、源码和归属材料，并单独报告冻结后未参与训练的评测结果。不得仅凭现有修复集成绩承诺整体质量，也不得仅凭代码 GPL 许可为权重补充授权。

[第三方说明](../THIRD_PARTY.md)包含会随下一次安装或升级复制的核心来源更正；本文件仅随源码提供。`fetch_assets.py` 保持历史发布资产和哈希，已有安装中的模型文档不会自动更新。
