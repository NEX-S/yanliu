# 语料注音的词组读音补充

`pinyin-phrases.json` 供 [`scripts/ime_lab.py`](../../scripts/ime_lab.py) 生成公开合成案例、[`scripts/build_ranking_corpus.py`](../../scripts/build_ranking_corpus.py) 构建公开排序语料时注音，不供输入法运行时使用。两个入口通过相邻的 [`pinyin_phrases.py`](../../scripts/pinyin_phrases.py) 检查固定版本 `pypinyin==0.55.0` 并加载同一份补充。它补充已确认缺失的十条词组读音，使用 `load_phrases_dict` 的原生词组格式，每字一个无声调音节，`ü` 写作 `v`。没有导入整部词库或按候选命中修改读音。

两个入口保留各自的注音边界。合成案例的完整多字目标已在 pypinyin 词组库中时，通过原生已分词列表接口优先采用目标自身的词组读音，避免前文分词跨过目标边界；其余目标仍与前文一起注音，再截取目标的音节。批量排序语料始终先注音原文中的完整连续汉字段，再截取目标的音节，因此目标只是词组的一部分时，也能使用整段的读音。

这十条读音逐项取自 [iDvel/rime-ice 的固定版本基础词库](https://github.com/iDvel/rime-ice/blob/da1fbe602e38f26db846fa10120ee64c2b0324c0/cn_dicts/base.dict.yaml)，来源 revision 为 `da1fbe602e38f26db846fa10120ee64c2b0324c0`，原文件 SHA-256 为 `0418d5103d8e40c8fff4b686a873ad03e9836cd0419ecae0b7e1e2526312dba0`。本仓库的 [资产 manifest](../manifest.json) 固定了同一文件、revision 和 [GPL-3.0 许可证](../dictionaries/LICENSE)。该词库声明汇集多个词表；这里的直接来源是此固定版 Rime Ice，不把上游汇集来源另称为本项目自有标注。本补充按 GPL-3.0 发布，保留 Rime Ice 署名与来源。

可在 `data/dictionaries/base.dict.yaml` 复核原始行（依次为词组、拼音、词频）：

```text
差旅	chai lv	8105
重构	chong gou	31315
重命名	chong ming ming	26535
代码行	dai ma hang	1380
去重	qu chong	9999
弹窗	tan chuang	13940
乐声	yue sheng	12000
认真地	ren zhen de	12007
轻轻地	qing qing de	123945
系安全带	ji an quan dai	5655
```

“系安全带”取自该文件第 183517 行，按完整词组补充，不修改单字“系”的默认读音。

冻结任一入口时，须同时保留相邻的 `scripts/pinyin_phrases.py` 和仓库根目录下同名相对路径的 `data/training/pinyin-phrases.json`；版本不符、文件缺失或结构无效会在开始生成前明确报错。有限与持续生成的 `summary.json` 分别以 `files.pinyin_phrases` 和 `files.pinyin_phrases_loader` 记录实际加载数据及加载器源码的 SHA-256，持续模式每批复核文件未变。批量排序语料的 `ranking-data-recipe.json` 以 `pinyin_phrases` 和 `pinyin_phrases_loader` 记录实际路径与 SHA-256，并与入口源码、原文和排除集一起检查文件未变。新增条目须逐项核对公开来源，不能从候选是否命中反推读音。

这项补充只影响此后生成的数据；历史训练集、评测集、快照和成绩保留原字节与原注音。

固定词组补充不能解决所有上下文歧义。例如 pypinyin 0.55.0 默认将“都会”读作城市含义的 `du hui`，会误注“我每次听这首歌都会听到哭”；全局改成 `dou hui` 又会损伤“国际都会”。即使仅补充“都会听”，也会把“在国际都会听音乐”跨词边界注错。[词组读音开发诊断](../eval/pronunciation-dev-v1.jsonl)同时保留副词、城市与跨词边界案例，供检查读音规则的副作用。

[注音修正案例](../eval/annotation-repairs-v1.jsonl)另存三条已确认的 Tatoeba DEV 副词误注，每条保留原拼音、原始行哈希、来源和署名；其中一句与词组读音诊断重复，不能重复计为独立样本。它们用于检查正确拼音下的候选，不构成对“都会”的全局读音规则。全局补充、历史评测文件与指标分母保持不变。

补充已修复 9 小时运行首轮核实的 12 次历史错误，并修复随后逐例复核发现的三次“地”误注：`他正在认真地`、前文 `认真` 接目标 `地`、`他轻轻地走` 中的 `di` 均应为 `de`。生成器对完整已知词组的边界保护同时修复了“逐步”接“调试”、“薄”接“薄雾”和“请关注”接“重构”时的跨界误分词。完整已知目标优先采用词典默认读音，是明确的词组先验；同一多音词的上下文语义可能需要其他读法，未收录目标也仍可能误注，因此自动注音须人工复核，不能将生成数据直接视为人工金标。
