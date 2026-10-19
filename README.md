# 砚流

Rust 中文拼音输入法，优先支持 GNOME / Wayland，通过 IBus 接入系统，运行时离线。本地词库负责切分、组合和有界候选召回；字符语言模型结合应用提供的前文，从召回结果中提升一个候选。默认 CPU 构建只带轻量依赖，显式启用的 `accelerator` feature 通过 OpenVINO 在 GPU 或 NPU 上执行同一模型。

目前按技术预览版使用。候选仍可能误排；随附的自动注音数据用于开发和回归，不代表自然用户准确率。默认下载固定的公共原版 4M 模型；[可选训练版 25M](data/model-cards/yanliu-25m-v1/README.md) 的权重与同源 IR 随 [v0.1.0 发布附件](https://github.com/NEX-S/yanliu/releases/tag/v0.1.0) 提供，公开训练数据与配方随源码保存。其他实验输出和个人学习状态不随源码分发。

## 构建与安装

需要 Rustup、系统 C 链接器和 Python 3.11 或更新版本；`rust-toolchain.toml` 固定 Rust 1.98.0、rustfmt 与 clippy。首次构建和获取资产需要网络，运行输入法不需要网络。先下载并校验资产，再构建 CLI：

```sh
git clone https://github.com/NEX-S/yanliu.git
cd yanliu
python3 scripts/fetch_assets.py
cargo build --locked --release --bin yanliu
target/release/yanliu decode nihao
target/release/yanliu --model data/models/sentence-model.safetensors decode xian --context 我去
```

CLI 不带 `--model` 时只使用词库。`fetch_assets.py` 拒绝哈希不符的下载，`--verify-only` 只校验本地文件。资产的固定来源、SHA-256 和许可证见 [data/manifest.json](data/manifest.json) 与 [data/THIRD_PARTY.md](data/THIRD_PARTY.md)。项目许可证为 GPL-3.0-or-later；词库、模型、vendor 与评测数据保留各自许可证。

IBus 服务通过 D-Bus 发布候选，由 GNOME Shell 显示面板；构建不需要 GTK 开发文件。桌面需要 IBus；协议测试还需要 Python GI 与 IBus typelib。在 Debian / Ubuntu 上可安装 `ibus python3-gi gir1.2-ibus-1.0`，使用虚拟环境运行开发脚本另需 `python3-venv`；Fedora 使用 `ibus ibus-libs python3-gobject-base`。本机验证环境为 Fedora 44 / GNOME Wayland，CI 使用 Ubuntu 24.04。安装到当前用户：

```sh
./packaging/install-local.sh
```

安装完成后重新登录，在 GNOME 设置 → 键盘 → 输入源中添加“砚流”。安装不会自动选择输入源。安装脚本为本机构建，产物应在目标机器重新编译。GPU/NPU 的资产准备、显式设备选择、可逆卸载和自定义前缀见 [安装说明](packaging/README.md)。

可选模型解包后，CPU 安装可设置 `YANLIU_MODEL_DIR=/absolute/path/to/yanliu-25m-v1`；GPU/NPU 使用同一模型目录及显式运行库。模型卡记录固定 SHA-256、来源、训练与评测限制。已有安装先按安装说明卸载；配置和个人学习保留，更换设备时须明确调整配置。

## 输入操作

- 未提交的拼音显示在候选面板，`▏` 标出拼音编辑位置；应用正文保持原样，确认提交后才替换应用中的选区。Esc 取消时不改正文。
- 空格选择当前候选，数字 1–9 选择当前页候选；部分选择只消费对应的原始拼音，保留余串。
- `Ctrl+H/L` 或上下方向键移动候选；`Ctrl+K/J` 或 `PageUp/PageDown` 翻页。候选横排显示。当前高亮候选的辅助说明区分补全、部分转换、换位纠错、邻键替换和漏键恢复，说明文字不进入提交内容。
- 左右方向键编辑拼音，Home/End 定位拼音首尾，Enter 提交拼音原文，Esc 取消输入。无对应候选的数字、无候选时的空格，以及 `0`，会先提交已有拼音原文，再透传该键，保留输入顺序。
- 大小写字母均留在预编辑中，拼音匹配忽略大小写，Enter 保留实际键入的大小写。英文、标识符、代码、URL 或命令使用 `Ctrl+Space` 切到英文模式，按键完整透传；切换时先提交未完成原文。
- 中文标点 `, . ! ? : ; [ ] < > ( )` 对应 `，。！？：；【】《》（）`，反斜杠输出顿号。单引号在已有拼音时分隔音节；其他 ASCII 标点保持半角。预编辑期间标点先提交当前候选，再提交余串原文及标点。

除上述引擎快捷键外，带 Ctrl、Alt 或 Super 的组合键透传。密码及敏感字段透传，不采集上下文或学习。

## 配置

安装创建 `~/.config/yanliu/config.json`；绝对 `XDG_CONFIG_HOME` 替换 `~/.config`。组件通过 `--config PATH` 在启动时读取一次，修改后需重启服务或重新登录。直接运行服务不自动读取用户配置。示例：

```json
{
  "data_dir": "/home/user/.local/share/yanliu",
  "learning_enabled": true
}
```

`model` 省略时使用有效 `data_dir/models/sentence-model.safetensors`。支持 `data_dir`、`model`、`user_dictionary`、`learning_enabled`、`device`、`runtime_dir` 和 `contextual_learning`；服务对应 `--data-dir`、`--model`、`--user-dictionary`、`--no-learning`、`--device`、`--runtime-dir`、`--contextual-learning`。`device` 默认为 `cpu`，显式 `gpu` / `npu` 必须配合 `runtime_dir` 和加速构建，初始化或执行失败保留原因报错，不自动切换设备。GPU 以 FP32 执行；NPU 使用设备默认精度，近似并列时排序可能与 CPU 不同。CPU 模式拒绝无用途的 `runtime_dir`。优先级为默认值 → 配置 → 显式 CLI；配置相对路径基于配置文件目录，CLI 相对路径基于工作目录，不展开 `~` 或环境变量。未知字段、非法类型、显式配置缺失或损坏均报出路径及原因。

## 候选与排序

组合 beam、完整候选和普通总候选预算固定为 12/24/32。另追加至多 8 个单次相邻字母换位、同排邻键替换或漏键恢复候选；单个字母、单独的 `zh/ch/sh` 声母片段和超过 24 字节的原始输入不追加纠错。三类纠错按上述顺序共享预算，来自词库真实读法，保留原始消费字节、操作位置、读音边界与静态分数。单引号约束音节边界；补出的未输入尾部始终是 Completion，不计完整转换，也不学习，同一种补全沿用一致的评分惩罚。漏键剩余槽优先消费更多原始字节，同消费量按静态分数及确定性插入顺序选择。

完整单音节另保留词典中所有同音单字符词条，供翻页选择；例如 `ji` 不会因为组合 beam 截断而选不到“吉”。这部分不扩大组合搜索，直接读取已加载的同音词条，保留真实消费量、来源和静态分数，因此单音节菜单可超过普通候选预算。`ji'` 仍完整消费原始 3 字节。首个音节以单引号明确分隔时，也保留该音节的全部同音单字符：例如 `ji'cheng` 可翻页选“吉”，消费 `ji'` 的 3 字节并保留 `cheng`，无需清空重输。内部音节边界不被忽略，也不会把没有分隔符的长串扩展为所有隐式前缀。学习与普通转换仍优先于纠错。

排序从至多九个完整候选中尝试提升一个候选，其余候选保持相对顺序。默认保护学习首选与无上下文的词库精确首选；部分候选和 Completion 不参与提升。词库精确词或明确选择处于首位时，静态分较弱的候选需要更多语言证据才能被提升，避免小模型的固有词形偏好轻易压过词频；组合路径的分数含分词成本，不用这项词频约束。无上下文时，语境学习开关不会单独解除公共语言模型的首选保护，个人条件分仍可纠正首选。Session 另保持普通完整转换优先于完整纠错。模型只改变顺序，不改变文本、来源、原始消费字节或静态分数。字符词表不能区分拉丁字母；实际评分组含“B站”这类中文词条时，语言先验整组中性，个人条件特征仍可学习。字符语言模型只评分窗口内的完整候选：窗口来自模型配置，默认 4M 模型最多 63 字符（另留首标记）。超长组的语言先验置为中性；个人条件特征仍读取完整文本，不把截断前缀当成完整候选评分。

## 学习

默认只保存明确选择所消费的拼音、文本和次数，不保存应用上下文或 Enter 提交的原文；Completion 可以选择，但未输入的补全部分不写入偏好。偏好最多 4,096 条，由后台线程合并、加锁并原子保存到 `$XDG_STATE_HOME/yanliu/learning.json`（`XDG_STATE_HOME` 未设置或非绝对时为 `~/.local/state/yanliu/learning.json`），目录 0700、文件 0600。正常退出保存最新状态，异常终止可能丢失最近的选择；加载或保存失败保留原始原因。`decode` 与 `evaluate` 不读写系统学习状态。

在配置中显式加入 `"contextual_learning": true` 并重启服务后启用语境学习。选词时，末 32 字前文、实际消费的原始输入和候选共同形成 4,096 维条件特征，只针对明确选中项上方、消费范围和纠错类别相同的竞争候选更新个人线性模型；部分选择只学习实际消费的前缀，敏感输入、Enter 与 Completion 不参与。排序融合冻结的语言分、个人条件分与温和的次数加分，同一拼音可随前文采用不同首选，已有次数不会永久锁定首选。参数在内存中更新，由同一后台线程与次数一起原子保存；不记录明文前文或候选日志，但参数仍可能记住私人信息，不视为匿名数据，不自动上传或分享。关闭 `contextual_learning` 后不应用也不更新已存参数，保留普通次数学习和已有参数。

配置 `"learning_enabled": false` 或服务参数 `--no-learning` 完全关闭学习：不读取已有偏好、不采集新选择、不启动持久线程或访问学习状态目录，选词照常；重新启用后恢复仍保留的偏好。

```sh
target/release/yanliu learning list
target/release/yanliu learning clear
```

`list` 输出状态路径和已持久化的拼音/文本/次数 JSON，不加载词库或模型；服务运行时可能尚有最多约 250ms 的待保存合并。`clear` 原子清空次数和个人参数，也可恢复损坏的快照。它与服务持有同一排他锁，服务使用该状态目录时会明确拒绝；先停止服务，清除后再启动，避免内存中的计数写回。

## 用户词表

用户词表是独立、由用户维护的 UTF-8 Rime 子集文件，默认不启用、不查找文件或目录。自行创建文件并保留词条的来源与适用许可，例如（词条各列之间用真正的 Tab）：

```text
---
name: my_words
version: "1"
...
砚流项目	yan liu xiang mu	100
离线工具	li xian gong ju
```

CLI 使用 `yanliu --user-dictionary /path/to/words.dict.yaml decode yanliuxiangmu`；IBus 在 `config.json` 加入 `"user_dictionary": "/path/to/words.dict.yaml"`，或使用同名 `--user-dictionary PATH` 参数。配置路径相对配置文件目录，CLI 路径相对工作目录；CLI 覆盖配置。删除配置字段并重启服务即可关闭；`null` 是非法类型。修改、删除文件中的词条后重启生效；程序只在启动时读取，不写回或热更新文件，文件缺失或损坏会保留路径和原因报错。

头部必须以独立的 `...` 行终止；随后允许空行、以 `#` 开头的注释，以及二或三列 TSV：文本、以空白分隔的小写 ASCII 拼音、可选非负有限权重（省略或空白列默认 1，兼容 `%` 后缀）。文本不得含控制字符。纯英文字母词条明确拒绝；英文通过切换输入模式透传。文件最多 8 MiB、10000 条有效数据行，每条拼接后的拼音最多 128 字节（不含音节分隔空白）；只有头部的空表合法。词条与主词库在启动时统一去重、建立索引，沿用普通候选来源、补全身份、消费量和静态预算，权重不保证首选。此词表与 Learning 选择偏好独立，`learning list/clear` 不查看或删除用户词表，关闭学习也不关闭显式词表。

## 开发与验证

单 Rust 包中，decoder/词库负责召回，Session 负责输入、光标、部分选择及 revision，model 负责排序，IBus 负责系统协议，CLI 负责离线诊断。IBus 先显示静态候选，再由共享后台线程更新排序，每个引擎只保留最新待执行任务；逐键路径不做磁盘 I/O。

候选质量和输入行为通过与 IBus 相同的 Session 路径在 CLI 中检查，不需要驱动真实 GNOME 窗口。`evaluate` 读取 JSONL，每行含 `input` 与 `gold`（多个答案用 `golds` 数组），可带 `id`、`context`、`bucket`：

```sh
mkdir -p results
target/release/yanliu --model data/models/sentence-model.safetensors evaluate --data data/eval/context-dev-v2.jsonl --output results/check.json
```

`--data -` 读标准输入，`--output -`（默认）写标准输出。命中要求完整消费原始输入且不是 Completion；未召回的目标仍计入分母，推理不读取 gold。随附数据集及其限制见 [data/eval](data/eval/README.md)。

`scripts/ime_lab.py` 通过 llmux 的 OpenAI 兼容接口请求固定模型 deepseek-flash 生成公开合成中文案例（前文、目标、领域），用固定版本 pypinyin 为目标注音，按规则校验后交给 `evaluate`。请求只发送抽象提示、明确提供的公开词对或已生成的目标列表，不发送密钥、私有语料、应用上下文或个人学习；输入法运行时不接入远端模型。

```sh
python3 -m venv .cache/yanliu-lab
.cache/yanliu-lab/bin/python -m pip install -r scripts/ime-lab-requirements.txt
OPENAI_BASE_URL="${ANTHROPIC_BASE_URL%/anthropic}/openai" \
  .cache/yanliu-lab/bin/python scripts/ime_lab.py --output-dir results/deepseek-run-001 --rounds 3 --cases-per-round 4
.cache/yanliu-lab/bin/python -m unittest discover -s tests -p test_ime_lab.py
```

本机 llmux 的 API base 为 `<origin>/openai`（请求 `/openai/chat/completions`），也可用 `--endpoint` 指定。输出目录须不存在，其中保存每轮原始请求与响应、`cases.jsonl`、CLI 报告 `check.json`，以及含拒绝原因和源码、binary、词库、模型哈希的 `summary.json`。教师内容无效的轮次记录后继续：全部有效退出 0，部分轮次无效退出 2，没有有效案例退出 1；HTTP 与 CLI 错误直接失败。案例是自动注音的合成开发数据，可能有多音字错误；用于调参后不能再当作独立评估。

训练语境消歧时可加 `--pairs data/training/homophone-pairs-v1.json`，每个公开词对生成两词各两条非空前文。程序先验证两词全拼相同，再拒绝额外目标、缺配对、不平衡和重复案例；`--cases-per-round 32 --rounds 15` 覆盖该文件的 120 对词。`--generate-only` 保存原始响应和通过校验的案例，跳过候选评分，供先冻结分组与未观测留出。来源与限制见 [训练数据](data/training/README.md)。

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

贡献时先复现具体输入或故障，按修改范围运行检查。候选问题优先提交能被 `evaluate` 读取的最小公开 JSONL 案例，注明期望候选与实际排名；不要提交应用上下文、私人词表、学习快照、密钥或本地对话记录。改动 vendor 或资产补丁时，同步固定来源与有效哈希，并在不带已下载资产的隔离副本中验证冷获取。

Agent 从 [AGENTS.md](AGENTS.md) 进入。
