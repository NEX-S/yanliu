# 开发指南

[← 返回首页](../README.md) · [使用手册](USER-GUIDE.md) · [安装与协议验证](../packaging/README.md)

以下命令均在仓库根目录执行；首次构建与资产获取见[快速开始](../README.md#快速开始)。

## 架构与 CLI

单 Rust 包中，decoder/词库负责召回，Session 负责输入、光标、部分选择及 revision，model 负责排序，IBus 负责系统协议，CLI 负责离线诊断。IBus 先显示静态候选，再由共享后台线程更新排序，每个引擎只保留最新待执行任务；逐键路径不做磁盘 I/O。

## CLI 评估

候选质量和输入行为通过与 IBus 相同的 Session 路径在 CLI 中检查，不需要驱动真实 GNOME 窗口。`evaluate` 读取 JSONL，每行含 `input` 与 `gold`（多个答案用 `golds` 数组），可带 `id`、`context`、`bucket`：

```sh
mkdir -p results
target/release/yanliu --model data/models/sentence-model.safetensors evaluate --data data/eval/context-dev-v2.jsonl --output results/check.json
```

`--data -` 读标准输入，`--output -`（默认）写标准输出。命中要求完整消费原始输入且不是 Completion；未召回的目标仍计入分母，推理不读取 gold。随附数据集及其限制见 [data/eval](../data/eval/README.md)。

## 生成公开合成案例

`scripts/ime_lab.py` 通过 llmux 的 OpenAI 兼容接口请求固定模型 deepseek-flash 生成公开合成中文案例（前文、目标、领域），用固定版本 pypinyin 为目标注音，按规则校验后交给 `evaluate`。请求只发送抽象提示、明确提供的公开词对或已生成的目标列表，不发送密钥、私有语料、应用上下文或个人学习；输入法运行时不接入远端模型。

```sh
python3 -m venv .cache/yanliu-lab
.cache/yanliu-lab/bin/python -m pip install -r scripts/ime-lab-requirements.txt
OPENAI_BASE_URL="${ANTHROPIC_BASE_URL%/anthropic}/openai" \
  .cache/yanliu-lab/bin/python scripts/ime_lab.py --output-dir results/deepseek-run-001 --rounds 3 --cases-per-round 4
.cache/yanliu-lab/bin/python -m unittest discover -s tests -p test_ime_lab.py
```

本机 llmux 的 API base 为 `<origin>/openai`（请求 `/openai/chat/completions`），也可用 `--endpoint` 指定。输出目录须不存在，其中保存每轮原始请求与响应。每轮校验完成即追加 `cases.jsonl` 并保存 `round-XXX.json` 中的接受项、拒绝原因与响应哈希；后续 HTTP 请求失败也会保留已完成的轮次。全部生成完成后才运行一次 CLI，保存 `check.json` 与含各轮记录和源码、binary、词库、模型哈希的 `summary.json`；中途失败不生成最终汇总。教师内容无效的轮次记录后继续：全部有效退出 0，部分轮次无效退出 2，没有有效案例退出 1；HTTP 与 CLI 错误直接失败。案例是自动注音的合成开发数据，可能有多音字错误；用于调参后不能再当作独立评估。

训练语境消歧时可加 `--pairs data/training/homophone-pairs-v1.json`，每个公开词对生成两词各两条非空前文。程序先验证两词全拼相同，再拒绝额外目标、缺配对、不平衡和重复案例；`--cases-per-round 32 --rounds 15` 覆盖该文件的 120 对词。`--generate-only` 保存原始响应和通过校验的案例，跳过候选评分，供先冻结分组与未观测留出。来源与限制见 [训练数据](../data/training/README.md)。

## 训练排序模型

`scripts/train_ranker.py` 用明确提供的公开 train/dev JSONL 和实际 CLI 候选做 CPU 监督训练，不读取系统学习，不向召回补入答案。训练分数包含当前词库静态差和首选提升门槛；每轮导出 f32/int8 并通过实际 CLI 比较开发集，按拼音相关组等权首选命中选择 checkpoint，包含未更新的 epoch 0，平局选更早的轮次。没有收益时保留原模型。

```sh
python3 -m venv .cache/yanliu-training
.cache/yanliu-training/bin/python -m pip install -r scripts/training-requirements.txt
.cache/yanliu-training/bin/python scripts/train_ranker.py \
  --model /path/to/initial.safetensors --binary target/release/yanliu \
  --train /path/to/frozen-train.jsonl --dev /path/to/frozen-dev.jsonl \
  --output-dir results/training-run-001
.cache/yanliu-training/bin/python -m unittest discover -s tests -p test_train_ranker.py
```

输出绑定初始模型、数据、当前源码、binary 和词库哈希，保留训练排除原因、完整开发分母、逐例收益与退化及候选身份检查。train/dev 的相同拼音或目标不得交叉，相关前文也应先归入同组；最终留出不传给训练命令。checkpoint 选择不自动安装模型，更换交付模型前还须检查未观测留出、已有回归、逐键成本和所选设备的同源 IR。

## 验证

按修改范围选择验证；真实词库测试需先获取资产：

```sh
cargo test --locked --all-targets --features ibus,accelerator
cargo test --locked --lib --features ibus real_dictionary -- --ignored --nocapture
cargo fmt --check
cargo clippy --locked --all-targets --features ibus,accelerator -- -D warnings
cargo build --locked --release --features ibus,accelerator --bins
YANLIU_BIN="$PWD/target/release/yanliu" /usr/bin/python3 -m unittest discover -s tests -p 'test_*cli.py'
/usr/bin/python3 -m unittest discover -s tests -p test_ibus_panel.py
/usr/bin/python3 scripts/test_ibus.py target/release/yanliu-ibus
/usr/bin/python3 tests/ibus_packaging.py
```

Python 协议测试需要系统 `ibus-daemon`、Python GI 与 IBus typelib，使用临时状态目录和私有 daemon，不切换桌面输入源。协议信号、CPU 回归与截图不能证明 Shell 绘制或键到屏幕延迟。

显式设备测试默认忽略，需要自行准备同源 IR、原生运行库和硬件；启用 `accelerator` 编译与运行其余测试不需要 OpenVINO 或 GPU/NPU。GitHub Actions 按上述 CPU 路径验证源码、真实词库、CLI、打包与私有 IBus 协议，不调用远端教师或访问桌面学习状态。

## 提交贡献

贡献时先复现具体输入或故障，按修改范围运行检查。候选问题优先提交能被 `evaluate` 读取的最小公开 JSONL 案例，注明期望候选与实际排名；不要提交应用上下文、私人词表、学习快照、密钥或本地对话记录。改动 vendor 或资产补丁时，同步固定来源与有效哈希，并在不带已下载资产的隔离副本中验证冷获取。

Agent 从 [AGENTS.md](../AGENTS.md) 进入。

## 候选与排序

组合 beam、完整候选和普通总候选预算固定为 12/24/32。另追加至多 8 个单次相邻字母换位、同排邻键替换或漏键恢复候选；单个字母、单独的 `zh/ch/sh` 声母片段和超过 24 字节的原始输入不追加纠错。三类纠错按上述顺序共享预算，来自词库真实读法，保留原始消费字节、操作位置、读音边界与静态分数。单引号约束音节边界；补出的未输入尾部始终是 Completion，不计完整转换，也不学习，同一种补全沿用一致的评分惩罚。漏键剩余槽优先消费更多原始字节，同消费量按静态分数及确定性插入顺序选择。

完整单音节另保留词典中所有同音单字符词条，供翻页选择；例如 `ji` 不会因为组合 beam 截断而选不到“吉”。这部分不扩大组合搜索，直接读取已加载的同音词条，保留真实消费量、来源和静态分数，因此单音节菜单可超过普通候选预算。`ji'` 仍完整消费原始 3 字节。首个音节以单引号明确分隔时，也保留该音节的全部同音单字符：例如 `ji'cheng` 可翻页选“吉”，消费 `ji'` 的 3 字节并保留 `cheng`，无需清空重输。内部音节边界不被忽略，也不会把没有分隔符的长串扩展为所有隐式前缀。学习与普通转换仍优先于纠错。

排序从至多九个完整候选中尝试提升一个候选，其余候选保持相对顺序。默认保护学习首选与无上下文的词库精确首选；部分候选和 Completion 不参与提升。词库精确词或明确选择处于首位时，静态分较弱的候选需要更多语言证据才能被提升，避免小模型的固有词形偏好轻易压过词频；组合路径的分数含分词成本，不用这项词频约束。无上下文时，语境学习开关不会单独解除公共语言模型的首选保护，个人条件分仍可纠正首选。Session 另保持普通完整转换优先于完整纠错。模型只改变顺序，不改变文本、来源、原始消费字节或静态分数。字符词表不能区分拉丁字母；实际评分组含“B站”这类中文词条时，语言先验整组中性，个人条件特征仍可学习。字符语言模型只评分窗口内的完整候选：窗口来自模型配置，默认 4M 模型最多 63 字符（另留首标记）。超长组的语言先验置为中性；个人条件特征仍读取完整文本，不把截断前缀当成完整候选评分。
