# 评测数据

以下候选评测 JSONL 可直接作为 `yanliu evaluate --data` 的输入；`input`、`context`、`gold`/`golds` 之外的字段只是来源记录，评测时忽略。gold 只用于评分。

- `context-dev-v2.jsonl`：24 条本项目自写的上下文消歧案例，例如前文“今年暑假我们准备去”后的 `xian` 应为“西安”。随仓库许可证（GPL-3.0-or-later）发布。
- `common-words-v1.jsonl`：26 条本项目手写的常用虚词、软件术语和 `kaiyuan` 同音词回归，包含“事件/时间”“集成/继承”以及“开源/开元”等相反语境。随 GPL-3.0-or-later 发布。案例已用于候选修复，模型仍可能误排；它们用于保留逐例收益与退化，不是独立留出或全对验收门。
- `replay-v3-dev.jsonl`、`replay-v3-test.jsonl`：由 Tatoeba 中文句子生成的 1000/2000 条整句案例，各一半以原句前缀为上下文。拼音由 pypinyin 对整句自动生成后再切分，可能有多音字错误，不是人工金标。按作者和近重复关系分组拆分，DEV/TEST 只有 186/352 个独立组，每组最多 20 条，条数不代表独立样本数或自然使用频率。构建时排除了此前使用过的评测文本和本地可读的 C4 匹配，完整 C4/LCCC 训练重叠未知。署名与许可（CC BY 2.0 FR）见 [replay-v3.NOTICE](replay-v3.NOTICE) 与[第三方说明](../THIRD_PARTY.md)，选择规则与哈希见 [replay-v3.lock.json](replay-v3.lock.json)。

这些数据都已被观测过：调参只用 `context-dev-v2` 与 `replay-v3-dev`，`replay-v3-test` 只作回归检查。它们不能证明自然输入准确率或独立泛化；新的泛化结论需要先锁定未观测的数据来源。

`learning-context.jsonl` 是 `yanliu learning replay --data` 的公开纠正序列，供检查“乘火车前往／此时此刻 + xian”的相反偏好与保存重载。它使用明确的 `select` 和原始消费字节，没有 gold 字段，不能交给 `evaluate`。案例随 GPL-3.0-or-later 发布；它们是开发回归，不保证任意语境都在相同次数内学会。命令见[学习回放](../../docs/DEVELOPMENT.md#回放个人学习)。

`replay-v3.lock.json` 是历史构建记录，其中 `input_hashes` 保留了当时用于排重的旧评测文件与本地语料哈希；这些输入和旧 builder 不随当前源码提供。仓库提供可直接运行的成品、逐句署名、上游来源和成品哈希，不承诺仅靠当前仓库完整重生成该历史划分。
