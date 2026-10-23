# 开发指南

[← 返回首页](../README.md) · [使用手册](USER-GUIDE.md) · [安装与协议验证](../packaging/README.md)

以下命令均在仓库根目录执行；首次构建与资产获取见[安装](../README.md#安装)。

## 架构与 CLI

单 Rust 包中，decoder 负责词库与拼音约束解码，Session 负责输入、光标、部分选择及 revision，model 提供语言概率与排序，IBus 负责系统协议，CLI 负责离线诊断。IBus 先显示静态候选，再由共享后台线程补充候选并更新排序，每个引擎只保留最新待执行任务；逐键路径不做磁盘 I/O。

CPU 推理将同层的两个搜索分支成批计算，让 INT8 矩阵共享权重读取与转换。各分支保留独立的位置和 KV，注意力逐分支计算；点积沿用原来的八路累加顺序。搜索在批次之间检查任务是否仍有效。

量化内核单输入按八个输出行分块，双输入按四行分块，每行的累加顺序相同。本机 Rust 1.98.0 的 native AVX2 实验中，四行单输入版本被编译为跨行权重拼装，显著慢于现有连续通道计算，因此保留当前分组。

尾部评分仍对完整词表计算归一化分母，只计算目标字的最终 log 概率。搜索和前缀恢复需要整份分布时，仍生成完整结果。两条路径共用相同的归一化统计，保留 f32 指数、f64 累加和减法的顺序。

排序复用上一轮候选的计算结果。有效前文和完整字符序列都相同时，直接使用完整累计分数；其他候选从已有状态覆盖的最长共享前缀续算。末字的概率由前一位置给出，因此普通候选评分只推进到倒数第二字，保留全部字的累计分数，省去暂时没有消费者的末字 Transformer 计算。后续搜索若需要这一步状态，就通过原有的 `advance` 路径补算。

搜索遇到相同的中文前缀时，复用已有 KV 和隐藏向量。搜索已算出的末尾概率分布随父状态保留；恢复到同一末尾时直接复用，恢复中间位置时仍从对应隐藏向量计算。搜索按原顺序逐字累计概率，最终 beam 所需的完整父状态交给本次排序，用父状态的分布给末字评分。缓存保留上一轮评分候选与本次搜索的这些父状态；本帧跳过评分时，下次搜索丢弃临时状态，前文窗口改变时全部清空。

### 输入与异步排序

IBus 按连接顺序处理焦点、内容类型属性和后续按键。引擎路径使用不并发派发的 Properties 接口，使密码类型通知在随后按键之前生效。修改输入后，Session 更新 revision、同步召回候选，IBus 先发布词库结果，再提交带有 generation 的快照。单个模型线程处理排序，待处理队列上限为 32，每个引擎只保留最新任务。

IBus 的异步运行时只开一个 worker；模型计算和学习持久化各用独立线程，不随机器的逻辑 CPU 数量扩张线程池。

结果返回时同时核对 generation 和 revision。焦点切换、导航、重置、敏感模式和销毁都会使旧任务失效。IBus 发送候选信号，由 GNOME Shell 绘制面板；拼音显示在面板的辅助文本中。

收到与当前有效窗口相同的前文时保留已显示菜单；前文改变时，从当前输入重新生成词库与个人偏好菜单，同步发布并重置候选高亮，再提交新快照，避免上一段前文补出的神经候选逐次累积或显示与提交不一致。输入、拼音光标和已明确学习的选词保持各自原有含义。

服务监听 D-Bus 连接结束，退出前停止排序队列并执行学习快照的最终 flush，释放学习锁，不留下占用同一状态目录的空闲服务。

### 学习与持久化

默认偏好最多保存 4,096 对实际消费的拼音、所选文本和次数。明确选择后先更新内存，独立持久线程约每 250 ms 合并突发更新，以原子替换保存快照；正常退出时 flush。Enter 提交的原文、Completion 未输入的部分和敏感字段不参与学习。

同一 Session 连续选择非补全的部分候选时，临时累计已消费的原始拼音和所选文本；选空余串后，除本段记录外再将完整组合记一次。单次完整选择不重复计数。增删原始拼音、有效外部前文改变、Enter、重置和 Completion 结束组合；相同有效前文回调、拼音光标移动、候选导航及模型重排保留组合。临时组合不进入后台快照或持久文件；关闭学习时不收集。完整组合只更新次数，不构造虚拟候选或额外训练个人参数。

Session 刷新时从已有内存计数中补充至多 8 条个人候选，来源为 User；与词库已有的文本及消费范围重合时保留词库候选，包括 Completion 身份。完整消费的普通转换与个人词条先于补全和纠错，使真实转换能够进入模型评分；只有补全与纠错时，保留二者原有静态次序。其后仍按显式分隔前缀、其他前缀分组，同组优先消费更多拼音，再按次数及静态分数排序，不增加持久索引或逐键磁盘读取。

`contextual_learning` 默认关闭。开启后，用有限前文和真实候选的纠正更新 4,096 维本地条件参数，与次数保存在同一快照中。不保存明文前文或候选日志；参数本身仍按私人数据管理。

个人分与公共语言模型的分值尺度不同。有个人反馈时，排序把当前短名单的语言和词库基础分、提升门槛同除以 `max(1, 基础分范围)`，再加个人分与次数项，避免大模型更宽的似然差压过纠正。没有反馈时沿用原分数。该调整保留既有特征、参数含义和快照格式，不需要清除选词记录。

### 回放个人学习

`learning replay` 在全新的隔离目录中重放公开选词，使用真实 Session 和持久化路径。每行含唯一 `id`、`input`、可选 `context`，需要纠正时提供明确的 `select.text` 和 `select.consumed`（原始拼音字节数）。连续分段选择用非空 `selects` 数组，与 `select` 互斥；数组各步在同一 Session 的余串菜单中选择，`consumed` 相对于当步剩余输入。两者都省略时只查询。

```json
{"id":"now-1","input":"xian","context":"此时此刻","select":{"text":"现","consumed":4}}
{"id":"segments-1","input":"niyouyaoxideyifuma","selects":[{"text":"你有要洗的衣服","consumed":16},{"text":"吗","consumed":2}]}
```

```sh
target/release/yanliu --model data/models/sentence-model.safetensors \
  learning replay --data data/eval/learning-context.jsonl \
  --state-dir results/new-learning-state --output results/new-learning-report.json
```

将 `--data` 改为 `data/eval/learning-readings.jsonl`，可以用完整词库回放“什锦”等非首选的纠正，并检查用品语境仍优先“湿巾”；每次回放使用新的状态目录和报告路径。

`data/eval/learning-boundaries.jsonl` 则检查词中间提交的纠正，例如“尤／其是我”“安／全理事”。它先查询全部前文，再各选词一次，最后查询原前文及同拼音的不同前文，便于区分“记住了这次选择”和“把所有前文都改成同一个词”。这些文件是固定开发诊断，不保证其他模型在相同次数内学会。

状态目录和报告都必须不存在，父目录须已存在；两者拒绝使用系统学习目录或其内部路径。命令不会读取桌面选词记录。回放固定开启语境学习，模拟系统配置 `contextual_learning: true` 的路径；普通次数学习也同时生效。默认配置关闭语境学习，不能用此回放结果推断它已在桌面启用。

个人上下文偏好只应用于明确选过的原始拼音，同一拼音在新前文中仍可调整；大小写和单引号分别记忆。其他拼音继续使用基础模型，已有的个人词条召回、部分选词和次数偏好照常生效。

报告保留每次查询前后的候选身份，以及语言、词库、个人参数与次数的评分贡献。旧 `select` 的 `selection` 字段保持原样；`selects` 对应的 `selections` 数组记录各步实际提交的文本、消费字节和剩余拼音，`after` 仍重新查询原整串。结束时 flush 并重新加载，比较完整候选与评分、快照字节；`reload_preserved_rankings` 和 `reload_preserved_snapshot` 表示结果。`promotion.index`、`promotion.order` 和评分项的 `index` 都引用排序前的 0 基候选下标，`order` 按总分列出已评分项；Session 保留首选门槛及受保护槽位后应用尾部顺序。`before_rank` 为 1 基名次。诊断报告含本次显式提供的明文前文，公开前需自行检查内容。

## CLI 评估

候选质量和输入行为通过与 IBus 相同的 Session 路径在 CLI 中检查，不需要驱动真实 GNOME 窗口。`evaluate` 读取 JSONL，每行含 `input` 与 `gold`（多个答案用 `golds` 数组），可带 `id`、`context`、`bucket`：

```sh
mkdir -p results
target/release/yanliu --model data/models/sentence-model.safetensors evaluate --data data/eval/context-dev-v2.jsonl --output results/check.json
```

`--data -` 读标准输入，`--output -`（默认）写标准输出。命中要求完整消费原始输入且不是 Completion；未召回的目标仍计入分母，推理不读取 gold。随附数据集及其限制见 [data/eval](../data/eval/README.md)。

## 生成公开合成案例

`scripts/ime_lab.py` 通过 llmux 的 OpenAI 兼容接口请求固定模型 deepseek-flash 生成公开合成中文案例（前文、目标、领域），用固定版本 pypinyin 和[小型公开词组读音补充](../data/training/pinyin-phrases.README.md)注音，按规则校验后交给 `evaluate`。完整多字目标若已收录在词组表中，优先使用目标自身的默认读音，避免“逐步／调试”被切成“步调”；其余目标先将前文与目标连起来，用固定 `jieba==0.42.1` 精确模式（`HMM=True`）分词后注音，再只截取目标的音节，保留“银／行”等跨边界读音。补充数据修复了“重构、弹窗、去重、重命名、乐声、差旅、代码行、认真地、轻轻地、系安全带”十条已核实的缺失读音。默认词组读音不保证适合所有多音词语境，自动注音仍须复核，不能按候选是否命中修改读音或筛掉案例。请求只发送抽象提示、明确提供的公开词对或已生成的目标列表，不发送密钥、私有语料、应用上下文或个人学习；输入法运行时不接入远端模型。

```sh
python3 -m venv .cache/yanliu-lab
.cache/yanliu-lab/bin/python -m pip install -r scripts/ime-lab-requirements.txt
OPENAI_BASE_URL="${ANTHROPIC_BASE_URL%/anthropic}/openai" \
  .cache/yanliu-lab/bin/python scripts/ime_lab.py --output-dir results/deepseek-run-001 --rounds 3 --cases-per-round 4
.cache/yanliu-lab/bin/python -m unittest discover -s tests -p test_ime_lab.py
```

本机 llmux 的 API base 为 `<origin>/openai`（请求 `/openai/chat/completions`），也可用 `--endpoint` 指定。输出目录须不存在，其中保存每轮原始请求与响应。每轮校验完成即追加 `cases.jsonl` 并保存 `round-XXX.json` 中的接受项、拒绝原因与响应哈希；后续 HTTP 请求失败也会保留已完成的轮次。全部生成完成后才运行一次 CLI，保存 `check.json` 与含各轮记录和生成器、读音补充、binary、词库、模型哈希的 `summary.json`；中途失败不生成最终汇总。教师内容无效的轮次记录后继续：全部有效退出 0，部分轮次无效退出 2，没有有效案例退出 1；HTTP 与 CLI 错误直接失败。案例是自动注音的合成开发数据，可能有多音字错误；用于调参后不能再当作独立评估。

持续评估按时长运行，每批生成后立即用真实 Session 评估，覆盖高频单字、常用同音词和计算机术语。例如连续运行 9 小时，每批 16 例，批次间隔 60 秒：

```sh
OPENAI_BASE_URL="${ANTHROPIC_BASE_URL%/anthropic}/openai" \
  .cache/yanliu-lab/bin/python scripts/ime_lab.py \
  --output-dir results/deepseek-continuous-001 --duration-hours 9 \
  --cases-per-round 16 --interval-seconds 60
```

`--duration-hours` 与 `--rounds`、`--pairs`、`--generate-only` 互斥。请求和 CLI 串行执行，分别受 `--request-timeout`、`--cli-timeout` 及剩余总时长约束；到截止时中断尚未完成的批次，不将其计入 `completed_rounds`。HTTP、CLI 或资产变化错误会明确终止，保留已完成批次及失败原因，不自动重试。持续运行前可冻结 binary、模型、词库与源码；`--binary`、`--model`、`--dictionary-dir` 指向所测快照，每批检查哈希未变。冻结生成器时，须同时复制相邻的 `scripts/pinyin_phrases.py`，并把 `data/training/pinyin-phrases.json` 按相对于 runner 仓库根目录的同名路径一起复制，保留其来源说明；pypinyin 或 jieba 版本不符、文件缺失或格式无效会在请求前明确报错。有限与持续模式分别以 `files.pinyin_phrases`、`files.pinyin_phrases_loader` 绑定实际加载数据和加载器源码的 SHA-256，持续模式每批也检查它们未变。两种模式的汇总均记录实际使用的固定 pypinyin、jieba 版本。

持续模式保存每批 `round-XXX-cases.jsonl`、`round-XXX-check.json` 和轮次记录，完成评估的案例追加到 `cases.jsonl`。`summary.json` 随批次更新，含起止时间、截止时间、完成批次、案例数、API 返回的 usage 和完整分母下的 top1/top5/recall、静态命中、rescue/breakage 计数。`issues.json` 按 `(input, context, golds, kind)` 去重，记录首次与最近出现、次数、样例路径及前十个完整消费的非 Completion 候选，完整候选身份保留在每批 check 中。

自动记录只标记疑似问题：目标未在返回候选中、排名靠后、模型破坏静态首选。未命中不能直接证明词库缺词，教师答案也未必唯一正确；多音字注音和前文歧义需核实后再加入回归集。候选检查不模拟明确选词和重启学习，单字母快捷键及个人偏好使用上面的 `learning replay`。

训练语境消歧时可加 `--pairs data/training/homophone-pairs-v1.json`，每个公开词对生成两词各两条非空前文。程序先验证两词全拼相同，再拒绝额外目标、缺配对、不平衡和重复案例；`--cases-per-round 32 --rounds 15` 覆盖该文件的 120 对词。`--generate-only` 保存原始响应和通过校验的案例，跳过候选评分，供先冻结分组与未观测留出。来源与限制见 [训练数据](../data/training/README.md)。

## 训练排序模型

`scripts/train_ranker.py` 用明确提供的 train/dev JSONL 和实际 CLI 候选做监督训练，默认 CPU/fp32，不读取系统学习，不向召回补入答案。训练分数包含当前词库静态差和首选提升门槛；每轮导出 f32/int8 并通过实际 CLI 比较开发集，按 `group_id`（未提供时按拼音）等权首选命中选择 checkpoint，包含未更新的 epoch 0，平局选更早的轮次。F32 输入的 epoch 0 是其实际 int8 导出；int8 输入另检查原模型与 epoch 0 的逐例结果一致。没有收益时保留原模型。

新导出权重的 `training_device` 和 `training_precision` 按本轮 recipe 的 `runtime` 覆盖，父 metadata 原样保留在 recipe 的 `model_metadata`，已有冻结资产不改写。

训练先用初始模型运行真实 CLI，将补充后的菜单、实际评分下标、词频项和首选提升门槛保存为 `train-source.json`。Python 直接使用 `evaluate` 的 `scoring.candidates` 与 `scoring.promotion`，不再复制 Rust 的短名单策略；神经搜索新增的错误组合也能成为真实负例。未召回的答案保留在状态统计中，不补进排名候选。缺失目标的后缀监督只用于 `gold_outside_scoring` 组：评分范围内至少有两个不同的完整候选，首选和至少另一个候选可由字表表示、语言分不全为零，且所有可接受目标均不在评分范围内。此时只保留完全在字表内、不含 ASCII 字母且字符数小于模型窗口的目标，以目标字的后缀负对数似然训练；前文按剩余窗口取尾部，只作条件输入，多个可接受目标等权。保护组、无竞争组和语言分中性组不进入这项监督；已有一个可接受目标参与评分时，也不为其余缺失目标追加监督。这类监督与真实候选的排序组分别计数和报告损失，不伪造 CLI 候选或评分下标。组合候选保留自己的语言分，不因来源而封顶，否则“落盘”这类词库未收录的合法组合无法超过“罗盘”。保留已正确案例并检查旧词是否退步；不以来源限制代替这项质量检查。

初始模型在完整 CLI 菜单中已经首选正确（`row.rank == 1`）的可训练排序组使用保留损失：`relu(margin + max(错误候选调整分) - max(可接受目标调整分))`。调整分仍包含真实词频项、首选原有的提升门槛，以及词表外候选与原首选持平的语言分；公式外加的 `margin` 是保护缓冲。因此，正确候选不是原首选时，它需要在未加首选门槛的分数上领先原首选错误候选至少两倍 `margin`。安全组的损失和梯度均为零，仍计入全部监督组的均值；多个可接受目标取最佳分，不把概率相加。初始首选错误的排序组继续使用可接受目标集合的交叉熵，缺失目标组继续使用后缀负对数似然。

开发集另运行无模型静态基线，检查父模型与每轮模型都保留静态普通候选的身份、消费字节和分数位；有真实完整读音时，补全或纠错路径可以被同文本、同消费量的普通转换替代。神经候选允许随权重改变，收益与退化仍相对父模型的实际排名计算。epoch 0 的回写检查比较实际候选、评分范围和决策，诊断浮点的微小差异不单独算行为变化。

提升前首项为拼接结果（Composed）时，最终排序把整串字符的对数概率之和除以该首项的字数，候选使用同一分母，以减少额外拆字稀释错误字符代价的情况。首项为词库词、个人词或纠错时保留每字平均分，避免压低“dian → 堤岸”这类正确的跨音节转换。这是依据开发案例选择的评分策略，不代表普适的概率单位修正。同字数候选分数不变；词表外候选仍跟随首项语言分，词频项和个人学习照常参与。排序训练采用相同规则；缺失目标的后缀监督仍取每字平均负对数似然。此项只改变最终提升评分，神经召回的搜索评分保持原有规则。

Transformer 使用完整前文，排序时词表投影只计算候选字位置，避免为不参与排序损失的前文生成 logits。可选普通文本批次在段落中全部有效目标位置计算父模型到学生模型的全词表 KL。

```sh
python3 -m venv .cache/yanliu-training
.cache/yanliu-training/bin/python -m pip install -r scripts/training-requirements.txt
.cache/yanliu-training/bin/python scripts/train_ranker.py \
  --model /path/to/initial.safetensors --binary target/release/yanliu \
  --train /path/to/frozen-train.jsonl --dev /path/to/frozen-dev.jsonl \
  --output-dir results/training-run-001
.cache/yanliu-training/bin/python -m unittest discover -s tests -p test_train_ranker.py
```

输出绑定初始模型、数据、当前源码、binary 和词库哈希，保留训练排除原因、完整开发分母、逐例收益与退化及候选身份检查。先按会话、生成响应及实质复制内容划分来源组，案例可用 `source_group` 标记：同源组与规范化的完整 `(input, context, gold)` 重复不能跨 train/dev。同一常用词在独立来源的新语境中可以跨 split；工具不会代替来源分组和长文本近复制清理。最终留出不传给训练命令。checkpoint 选择不自动安装模型；更换交付模型前检查已确认缺陷、已有回归和逐键成本。缺陷修复可以使用已观测案例，但必须披露重叠，不能把修复集成绩称为独立泛化；新的泛化结论需要未观测留出。私人对话及其派生模型保留在本地，不能因项目开源而自动公开。

### 从随机权重训练

`scripts/pretrain_model.py` 对每行独立的 UTF-8 段落做下一字符预训练。`--spec` 指定 JSON 文件，其中 `config` 包含 `vocab/n_layer/n_head/n_embd/context/dropout`，`vocab` 是以 `<pad>/<unk>/<bos>` 开头的完整字符表，`provenance` 声明 `license/data_classification/attribution` 及实际数据来源。全部参数随机初始化。字符表应在训练前核对常用字覆盖，开发文本须来自独立来源组；按完整开发集 CE 保留最佳 F32 权重。默认最多 12 遍、3 亿有效目标字符或 6 小时，先到即停，开发 CE 连续两次不改善也会停止；这些预算可通过参数调整。

`--batch` 表示 64 字符窗口的等效位置预算，例如 `--batch 16` 每次更新最多处理 1024 个输入位置。编码仍按独立段落加 BOS、每 64 个目标切块，不跨段拼接；训练时按每块非 PAD 目标跨度 1..64 分组，每组打乱后按 `floor(batch*64/跨度)` 条切批，再打乱批次顺序，每个完整 epoch 恰好覆盖每块一次。张量裁到当前批次的实际跨度，短段无需补到 64；开发集按相同长度分组完整评估。UNK 保留在输入中，PAD/UNK 目标忽略，`--max-tokens` 精确限制有效目标数。学习率按实际累计有效目标字符推进，预热长度为计划有效目标的 5% 与 100 次位置预算的较小值，随后余弦降到峰值的 10%；峰值 `--lr` 不随 batch 自动放大。这些含义与实际预算均写入 recipe。

Intel GPU 需要单独的 XPU 环境，按 [PyTorch Intel GPU 指南](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)安装对应构建，并安装 `numpy==2.5.4`、`safetensors==0.8.0`；不要把上述 CPU requirements 安装进 XPU 环境。CPU 训练使用显式 fp32，XPU/bf16 使用 MATH SDPA，避免本机 fused BF16 attention 曾出现的非有限值；训练检查 loss 与梯度是否有限。请求 XPU 而设备不可用时直接报错，不回退 CPU。当前 NPU 路径用于推理。

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /path/to/xpu-env/bin/python scripts/pretrain_model.py \
  --train /path/to/text-train.txt --dev /path/to/text-dev.txt \
  --spec /path/to/model-spec.json \
  --output-dir results/pretraining-run-001 \
  --device xpu --precision bf16 --batch 16 --lr 3e-4 \
  --epochs 12 --max-tokens 300000000 --max-seconds 21600

ONEAPI_DEVICE_SELECTOR=level_zero:gpu /path/to/xpu-env/bin/python scripts/train_ranker.py \
  --model results/pretraining-run-001/best-f32.safetensors \
  --binary target/release/yanliu \
  --train /path/to/ranking-train.jsonl --dev /path/to/ranking-dev.jsonl \
  --text-train /path/to/text-train.txt --text-batch 16 \
  --device xpu --precision bf16 --epochs 2 --lr 1e-5 \
  --output-dir results/ranking-run-001
```

本机 XPU 环境的默认 SYCL 设备枚举会加载 PoCL 及系统 LLVM21，随后导入 Triton 时发生 LLVM 符号冲突，导致排序器测试进程崩溃。上面的 GPU 训练命令限定使用 Level Zero GPU；若用同一 XPU 环境运行 CPU 张量测试，可仅对该进程排除 SYCL 设备，避免无用的 OpenCL 枚举。选择器语法见 [Intel 文档](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/intel-oneapi-level-zero-switch.html)。

```sh
ONEAPI_DEVICE_SELECTOR='!*:*' /path/to/xpu-env/bin/python -B \
  -m unittest discover -s tests -p test_train_ranker.py -v
```

本机在 `level_zero:gpu` 和 `!*:*` 两种选择器下的完整排序器测试均通过，无需预加载 Triton。设置只作用于命令启动的进程；使用前述 CPU requirements 环境时，沿用普通测试命令。

排序时 `--text-train` 为每次更新加入一批 `KL(冻结初始父模型 || 学生模型)`，温度为 1，与监督组均值的系数均为 1。教师直接深拷贝初始模型，在 eval 模式、关闭参数梯度和 `no_grad` 下前向计算；学生保持原有训练模式、dropout 和随机数顺序。文本仍服从相同来源拆分，每个已知字符只作一次有效目标，不跨段拼接；PAD 和 UNK 目标忽略，UNK 仍保留在输入上下文中。KL 按有效目标位置取均值，覆盖完整词表，报告字段为 `text_teacher_kl`；不再叠加普通文本 CE。recipe 分别记录源字符数、有效目标数、未知字符和 padding，以及教师目标、设备、精度和来源哈希。从零训练产物的归因来自 spec 中的实际来源声明。

### 准备公共中文语料

`scripts/prepare_general_corpus.py` 使用固定版本的 LCCC 日常对话、中文 Wikipedia、COIG-CQIA 人类回答、TNEWS 标题，以及新增的 NaturalConv、CPED 纯文本对话。原始文件须先放到 `data/cache/retrain-20261021/sources/`，文件名、上游版本、URL 与哈希见脚本中的 `SOURCE_FILES`；同目录的 `1.txt/2.txt/3.txt` 分别提供 3500、3000、1605 个规范汉字，可从[固定版本的字表抄录](https://gist.github.com/z-rui/4cb3431f2bcf26ea39cd569a58e66003/5c4510138344632327abcf966195aa8314e5c84b)获取。数据准备使用独立环境：

```sh
python3 -m venv .cache/yanliu-corpus
.cache/yanliu-corpus/bin/python -m pip install -r scripts/corpus-requirements.txt
.cache/yanliu-corpus/bin/python scripts/prepare_general_corpus.py \
  --output data/cache/retrain-20261021/corpus \
  --evidence results/retrain-20261021/data-preparation \
  --previous-train data/cache/general-model-20261021/corpus/train.txt \
  --exclude data/cache/general-model-20261021/ranking-holdout.jsonl
.cache/yanliu-corpus/bin/python scripts/build_ranking_corpus.py \
  --documents data/cache/retrain-20261021/corpus/documents.jsonl \
  --output-dir data/cache/retrain-20261021 \
  --evidence-dir results/retrain-20261021 \
  --exclude data/cache/general-model-20261021/ranking-holdout.jsonl
```

准备程序在创建输出前验证每个原始数据文件的固定 SHA-256。新 `data-recipe.json` 与模型 spec 的 `provenance.source_attributions` 均保留各来源声明；预训练将整个 provenance 写入权重 metadata，后续排序微调继承它。NaturalConv 记录固定 LICENSE 的 URL、哈希、非商业用途与分发限制，训练权重是否属于受限衍生作品仍需单独确认；COIG-CQIA 和 TNEWS 的许可缺口仍明确保留。历史发布包与冻结配方保持原字节，补充说明见[来源复核](../data/model-cards/SOURCE-REVIEW.md)。

准备程序先分配来源大组，再清洗、规范化去重与排除已有公开评测内容。LCCC、NaturalConv、CPED 逐句清洗后按原轮序连接，补足缺少的句末标点，再限长切段；不跨会话连接。CQIA 仅收人类正文，同一来源域的问题共用 split、保留全部不同回答，再按正文去重，不收小红书营销源。NaturalConv 同一 `document_id` 的所有会话共用 80/10/10 的稳定 hash 划分，研究已见的 0..6 号新闻整体排除；CPED 同一 `TV_ID` 共用官方 26/5/9 部剧的 train/dev/holdout 划分，`Dialogue_ID` 只界定具体对话。`documents.jsonl` 的 `group` 表示具体会话/页面/问题，`split_group` 表示不能跨 split 的相关来源组，新增来源分配冻结到 `source-groups.json`。新增 dev/holdout 还按相同 exact/near 正文规则排除旧 `train.txt` 的重复，旧训练文件只用于防污染，不排除新训练内容的合理复用。

训练预算默认分别为 LCCC 15M、Wikipedia 25M、CQIA 最多 12M、NaturalConv 最多 2M、CPED 全训练集（1M 上限）、TNEWS 1M 字符，不足如实记录。NaturalConv 的训练抽样审查发现同音错字和表达问题，因此只作少量口语补充，不自动改写原文，也不将抽样错误比例推为总体错误率。训练文本保留中文标点；词表包含全部 8105 规范汉字、词库单字、ASCII、固定标点和训练集高频扩展字符。`data-recipe.json` 保留来源哈希、质量备注、规则、来源数量、实际训练频次覆盖与 OOV。`compact-spec.json`（6 层×192、4 头）、`small-spec.json`（6 层×256、4 头）、`large-spec.json`（8 层×448、8 头）共享字表和 64 字符窗口，可直接传给预训练的 `--spec`。

排序案例继承原文档 split，先按固定 hash 和各来源独立预算选样，再对完整连续汉字段注音后截取目标。`build_ranking_corpus.py` 与合成案例入口共用 `pinyin_phrases.py`，检查 `pypinyin==0.55.0`、`jieba==0.42.1` 并加载上述十条公开词组读音。先用 jieba 精确模式（`HMM=True`）给完整汉字段分词，再把每个词作为字符串交给 pypinyin 注音，保留复合长词内部的已有短词读音；同十条补充也作为 jieba 词界，不增添按评测答案编写的词组。整段对齐、目标切片的规则保持不变，词内目标仍从整段结果取音节。合成入口保留已知完整目标词优先的语义。自动分词和词典读音不能保证语境中的多音字正确。配方同时记录两个工具的版本。`ranking-data-recipe.json` 的 `pinyin_phrases`、`pinyin_phrases_loader` 记录实际路径和 SHA-256，并与入口源码、原文和排除集一起检查文件未变。版本不符或补充文件无效时，不开始生成；历史数据与成绩保持原样。需要上下文的案例保持 hash 选定的任意字界和长度，保留目标在同一段落中之前最多 32 个字符，包括标点与前轮话语；已有前文时，目标可从当前汉字段首字开始。hash 为每个文档长度槽位以 25% 概率请求无上下文；此时目标必须是长度落在该槽位区间的完整汉字段，左右均须是标点、空白或文档首尾，不从字段内部随机截取。文档内部段界检查相邻段落的实际边缘字符，不能把无标点的强制切段或数字、拉丁字母造成的汉字段边界当作自然边界；无可用完整字段的槽位按边界或长度拒绝原因计数，不改成有上下文或补采样。无上下文仍为空串，不跨不同段落或会话取前文。默认 train 预算 80,000、dev 2,000；fresh holdout 仅收 NaturalConv 最多 1,000 和 CPED 最多 750 例，旧来源 heldout 仅作语料回归。可重复使用 `--train-limit SOURCE=COUNT`、`--dev-limit SOURCE=COUNT`、`--holdout-limit SOURCE=COUNT` 改固定预算。`source_group`/`group_id` 保留真实 split 大组，`document_group` 表示具体采样文档；CPED holdout 最多只有官方 9 部剧的独立大组。程序不调用模型、不按召回或成绩挑样，实际数量不足时如实记录。原文错字及自动拼音可能影响监督，因此这些案例不是干净的人工金标。`ranking-dev.jsonl` 用于选模；`ranking-holdout.jsonl` 和 `corpus/holdout.txt` 保留至训练与策略冻结后评估，不传给训练入口。

语料准备与排序案例构建都默认排除仓库内 `data/eval/*.jsonl`；额外已观测评测须分别向两个命令重复传入 `--exclude PATH`，显式指定的文件必须存在，路径和 SHA-256 进入各自配方。语料按规范化后至少 6 字符的 context+gold 作为子串排除，排序案例按 NFKC/casefold/字母数字规范化后的 `(context, gold)` 去重和排除，拼音修正不会让相同中文样本绕过旧评测排除；不同实质前文仍可保留，不把短常用词整体禁用；不会自动扫描本地 `results/` 或个人数据。

### 7M 实验配置（2026-10-21）

本轮从随机权重训练的字符模型为 6 层、4 个注意力头、隐藏维度 256，词表 9,506 项，窗口 64 字符，约 719 万参数。INT8 源权重约 7.14 MiB。词表覆盖全部 8,105 个规范汉字，并分别记录 3,500 个一级常用字的覆盖、正文出现次数与 OOV 比例。

这是历史 7M 实验，当前默认已更新为从随机权重训练的 4M v2，可选 25M v2，见[模型说明](../README.md#模型)。默认 CPU 直接使用 Rust 推理与前缀缓存；GPU / NPU 使用显式准备的固定形状 IR。Session 回放用于检查处理成本，不包含 GNOME 绘制或真实键到屏幕延迟。

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

词库解码的组合 beam、完整候选和普通总候选预算为 12/24/32。另追加至多 8 个单次相邻字母换位、同排邻键替换或漏键恢复候选；单个字母、单独的 `zh/ch/sh` 声母片段和超过 24 字节的原始输入不追加纠错。三类纠错按上述顺序共享预算，来自词库真实读法，保留原始消费字节、操作位置、读音边界与静态分数。单引号约束音节边界；补出的未输入尾部始终是 Completion，不计完整转换，也不学习，同一种补全沿用一致的评分惩罚。漏键剩余槽优先消费更多原始字节，同消费量按静态分数及确定性插入顺序选择。

完整单音节另保留词典中所有同音单字符词条，供翻页选择；例如 `ji` 不会因为组合 beam 截断而选不到“吉”。这部分不扩大组合搜索，直接读取已加载的同音词条，保留真实消费量、来源和静态分数，因此单音节菜单可超过普通候选预算。`ji'` 仍完整消费原始 3 字节。首个音节以单引号明确分隔时，也保留该音节的全部同音单字符：例如 `ji'cheng` 可翻页选“吉”，消费 `ji'` 的 3 字节并保留 `cheng`，无需清空重输。内部音节边界不被忽略，也不会把没有分隔符的长串扩展为所有隐式前缀。学习与普通转换仍优先于纠错。

CPU 排序保留至多十六项完整候选的短名单，最后两项优先保留不同长度的文本，另比较按原候选顺序排列的前 64 个基本汉字单字。单字概率直接取自已算好的前文分布，不为每个字再运行一次 Transformer。固定 IR 后端仍使用九项短名单。首选仍由提升门槛决定；确定首选后，其余已评分的完整候选在原有合格槽位内按总分稳定降序排列，同分沿用原顺序。此步不增加模型推理，未评分候选、部分候选、Completion 和受保护的个人候选保留槽位。评分、显示与训练均遵守 Session 的普通转换／纠错分组。

CPU 后台还会进行有界神经召回。真实单字词典行构成拼音 DAG，显式单引号限制边界，每个跨度最多保留 64 个模型可表示的单字；反向移除无法完整消费输入的路径。搜索按输出字数分层，每层先从前文分布读取合法后继的概率，再仅为保留的十二条路径中的非终点运行增量推理。终点按平均每字 log 概率保留至多十二项，静态分只作同分规则。超出模型完整窗口的目标不参与搜索；单音节单字已由静态路径全部保留，也无需搜索。过期任务在层间和每次推理前取消。

神经路径只增加完整转换，其 `score` 仍来自真实单字词频的原始拼接公式。与已有普通转换重合时保留已有身份；完整的原样读音可以替代同文本、同消费量的补全或纠错路径。合并后重新应用内存中的学习次数和普通／纠错分组；新路径另加入评分短名单，避免再次被十六项上限挡住。精确词、简称、部分候选、补全和纠错仍由原词跨度路径提供。CLI 的 decode、evaluate、学习回放和 IBus 调用同一个 `Model::refine`，IBus 复用原模型线程与有界队列。

默认保护学习首选与无上下文的词库精确首选。词库精确词或明确选择处于首位时，静态分较弱的候选需要更多语言证据才能被提升，避免模型的固有词形偏好轻易压过词频；组合首选的分数含分词成本，不用这项词频约束。组合候选本身仍可凭上下文胜出。无上下文时，语境学习开关不会单独解除公共语言模型的首选保护，个人条件分仍可纠正首选。

Session 保持普通完整转换优先于完整纠错。排序阶段只改变顺序，不改变候选文本、来源、原始消费字节或静态分数。实际评分组含“B站”这类带拉丁字母的词条时，语言先验整组置为中性，个人条件特征仍可学习。字符语言模型只评分窗口内的完整候选：窗口来自模型配置，默认 4M 模型最多 63 字符，另留首标记。超长组的语言先验置为中性；个人条件特征仍读取完整文本，不把截断前缀当成完整候选评分。

词表外候选的语言分与当前首选持平，只保留静态词频和个人选择带来的区别，避免共享 UNK 概率错误提升生僻字。首选本身含词表外字符，或可完整表示的候选不足两个时，整组语言先验为中性。明确选择这些候选仍参与本地偏好学习。
