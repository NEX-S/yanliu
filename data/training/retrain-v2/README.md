# 砚流 v2 随机训练记录

本轮从随机参数训练默认 [yanliu-4m-v2](../../model-cards/yanliu-4m-v2/README.md) 和可选 [yanliu-25m-v2](../../model-cards/yanliu-25m-v2/README.md)。两者的来源 URL、数据与源码哈希、实际参数、阶段权重和指标分别保留在模型卡的 [4M provenance](../../model-cards/yanliu-4m-v2/provenance.json) 与 [25M provenance](../../model-cards/yanliu-25m-v2/provenance.json)。仓库保留文本记录，权重由模型包携带；本目录不复制语料或学习状态。其他开发约束见[开发说明](../../../docs/DEVELOPMENT.md)。

数据入口是 [`prepare_general_corpus.py`](../../../scripts/prepare_general_corpus.py)：按 provenance 的固定来源下载并核对原始文件，准备三份标准字表和字典，按会话/页面/问题/新闻/TV_ID 分组，再清洗、拆分和去重。用 `--root`、`--output`、`--evidence` 和必需的 `--previous-train` 指定输入输出；旧训练文本及额外 `--exclude` 应与记录具有相同 SHA，路径可按新工作目录指定。随后 [`build_ranking_corpus.py`](../../../scripts/build_ranking_corpus.py) 从同一 `documents.jsonl` 派生拼音样本，继承分组、固定抽样池并排除既有评测。数据处理使用 OpenCC 1.4.2 的 t2s 和 pypinyin 0.55.0，另需 pyarrow；原始材料的已知许可和未知说明见模型 NOTICE。这里没有自动下载或替代缺失输入的入口。

[`pretrain_model.py`](../../../scripts/pretrain_model.py) 读取数据入口生成的 spec、train 和 dev，从随机参数进行下一字符预训练。4M 为 6 层/4 heads/192 hidden，25M 为 8 层/8 heads/448 hidden；两者 V=9358、context64、dropout0.1、tied embeddings。实际运行均为 seed20261021、AdamW、lr0.0003、batch64 等价位置预算4096、400M 有效目标上限，FP32 主参数/BF16 运算，按文本 dev CE 选模。实际参数位于 provenance 的 `training.pretraining.arguments` 与 `optimizer`，数据规则和 spec 哈希同处记录。

[`train_ranker.py`](../../../scripts/train_ranker.py) 使用上述选中 FP32 权重、明确的 CLI binary、80,000 条排名训练和 1,917 条 dev，另读普通文本训练。实际为 1 轮、batch16、text-batch16、lr0.00001；53,329 条样本参与排序 CE，文本 CE 系数为1。它导出 FP32/逐行 INT8，并按实际 INT8 CLI dev 组等权 top1、平局较早，读取本次 selection 决定发布文件，不能强制指定 epoch1。当前源码的 CPU 依赖见 [`training-requirements.txt`](../../../scripts/training-requirements.txt)；历史运行使用 PyTorch 2.14.1+xpu，设备和 precision 参数均已记录，不静默替换训练后端。

这些入口和记录支持重建数据与训练过程，但不是保证重新生成同一权重字节的封装。缺失旧训练/排除输入、依赖或字表时应先补齐并核对；不要改换输入后沿用原哈希。训练采用非确定性算法，路径、源码、设备实现和 safetensors 序列化也可能改变字节；新运行须保存自己的 recipe、selection 与哈希。开发集用于选模；冻结默认决定后才执行单次全量 holdout，没有根据其结果调整权重、阈值或选择。
