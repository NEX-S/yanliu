# yanliu-25m-v1 冻结训练输入

本目录保留 2026-10-19 最终 25M 微调实际使用的数据与固定配方。自写数据、配方及复现代码随项目按 [GPL-3.0-or-later](../../../LICENSE) 发布；模型权重及其微调贡献按 Apache-2.0 发布，见 [模型卡](../../model-cards/yanliu-25m-v1/README.md)。

`train.jsonl` 为 312 条公开合成中文语境：180 条新拼音训练案例，以及 132 条已知公开同音词语境。后 132 条最初被划为 known-regression，随后明确用于本次 targeted curriculum；它们上的改善只表示训练拟合。312 条中实际短名单含目标且可训练的为 298 条，其余 14 条目标在短名单外，不加入损失，不把目标补入候选。

`novel-dev.jsonl` 为 44 条、11 个相关组的新拼音开发案例，与训练的拼音及目标不重叠。`mixed-dev.jsonl` 保留这 44 条及另外 50 条已知词的新语境，共 94 条、45 组；它是已观测的混合开发集，其中已知词的输入和目标可以与训练重叠，不能传给拒绝 train/dev 拼音或目标重叠的训练入口。

合成案例由开发脚本通过 llmux 调用固定教师 `deepseek-flash`，围绕仓库公开词对生成；目标用 `pypinyin==0.55.0` 自动注音。混合开发集另外包含自写的公开诊断语境。所有文件只含公开目标与合成/自写前文，不含真实对话、应用上下文或系统学习。原始网关请求与响应不随本目录发布。自动注音和教师语境未获人工金标保证，不代表自然输入分布或准确率。

原始分组将相同全拼、相同目标、相同规范化前文，或长度至少 12 且 SequenceMatcher 相似度至少 0.9 的前文连为相关组；新组按固定哈希划分。原先的 32 条留出、已有 common/context/replay 数据均已被观测，后续只作回归，不能重新声明为独立留出。此目录不把这些数据加入训练或选模。

## 实际训练与选模

起点是固定修订的公共原版 [metasequoiaime/pinyin-ime-reranker-25M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M/tree/75295b373a7ef0d14e9bebd32c1625c9cfcfea49)，原文件 SHA-256 为 `0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469`。仅把 context 从 128 改为 64 并保留 `pos.weight` 前 64 行，其余张量不变，历史派生文件 SHA-256 为 `4d4bc10ee9e1d0ca82f2bad3a0ee7c7ba598dc1fcb0dc79ef267914a40473448`。

固定运行 3 轮 CPU 监督训练：AdamW，学习率 `1e-5`，betas `(0.9, 0.95)`，weight decay `0.01`，梯度范数上限 `1.0`，batch `4`，threads `2`，seed `20261019`，共 225 个优化步骤。训练目标使用真实 CLI 短名单、逐字平均语言分、直接词频差系数 `2` 和首选提升门槛 `0.25`。没有扩大召回或改变候选身份。

原训练脚本在 novel-dev44 上比较 epoch 0–3，按相关组等权 top1、平局选较早轮次，实际选出 **epoch 0**。实验事先另行固定了 mixed-dev94 的最终选择：训练完成后用真实 CLI 比较 epoch 0–3，按同一组等权准则选 **epoch 3**。其组等权值依次为 `149/180`、`437/540`、`451/540`、`467/540`，逐例 top1 分别为 `73/94`、`73/94`、`76/94`、`77/94`。这些是选模开发结果，不能当作独立泛化或自然用户准确率。两套选择原样保留在 `recipe.json`，不把训练脚本的选择改写成 epoch 3。

## 重训流程

从仓库根目录执行。先按根 README 获取固定资产并构建 CLI，再安装训练依赖；不需要调用教师重新生成数据。

```sh
python3 scripts/fetch_assets.py
cargo build --locked --release --bin yanliu
python3 -m venv .cache/yanliu-training
.cache/yanliu-training/bin/python -m pip install -r scripts/training-requirements.txt
mkdir -p .cache/yanliu-25m-source
curl --fail --location \
  https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M/resolve/75295b373a7ef0d14e9bebd32c1625c9cfcfea49/sentence-model.safetensors \
  --output .cache/yanliu-25m-source/upstream.safetensors
.cache/yanliu-training/bin/python data/training/yanliu-25m-v1/reproduce.py \
  --upstream-model .cache/yanliu-25m-source/upstream.safetensors \
  --binary target/release/yanliu \
  --output-dir results/yanliu-25m-reproduction
```

输出目录必须不存在。入口校验上游权重、冻结数据与词库哈希，裁剪位置表，调用现有训练脚本，再独立执行 mixed-dev94 选模；最终写出 `selection.json` 和 `selected.safetensors`。它保留训练脚本自己的 novel-dev 选择，实际得到的选模轮次由当前运行决定，不硬编码为 3。IR 可再通过 `scripts/export_ranker.py` 从选中源权重导出。

此流程可重做训练与行为选择，**不保证逐字节再生成发布文件**。历史训练配方绑定当时的绝对输出路径、CLI binary 和完整源码哈希，其哈希被写入权重 metadata；公开源码树、构建与工作目录变化会改变该绑定，safetensors 头的序列化也可能改变文件字节。历史来源绑定、数据哈希、指标和最终发布哈希保留在 `recipe.json` 与模型 provenance 中，供核对实际运行，而非承诺一个现成训练命令必然直接选出 epoch 3 或产生相同 SHA-256。
