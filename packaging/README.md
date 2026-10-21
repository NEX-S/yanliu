# GNOME 本地安装

先按主 README 获取并校验资产，再执行：

```sh
./packaging/install-local.sh
```

默认安装以 `-C target-cpu=native` 构建 CPU release 服务；产物应在目标机器重新编译。默认前缀为 `~/.local`，也可设置绝对、shell 安全的 `YANLIU_PREFIX`。实际二进制、词库和模型位于 `share/yanliu`，`bin/yanliu-ibus` 是指向实际二进制的符号链接，组件 XML 位于 `share/ibus/component/yanliu.xml`。默认安装不包含训练、评测文件、相邻 IR 或原生 runtime；显式 GPU/NPU 安装见下文。

CPU 默认安装仓库的 `data/models/sentence-model.safetensors`，即[从随机权重训练的 4M v2](../data/model-cards/yanliu-4m-v2/README.md)；设置 `YANLIU_MODEL_DIR=/absolute/model/directory` 可安装选定目录中的同名源权重，例如[可选 25M v2](../data/model-cards/yanliu-25m-v2/README.md)。模型目录内的 `LICENSE`、`NOTICE`、`README.md` 与 `provenance.json` 随发布包提供，也随模型复制并纳入安装哈希。v2 发布包供 CPU 使用，不包含 IR 或原生加速运行库。

`installation.json` 记录实际选中的源权重路径、SHA-256 和全部安装文件的 SHA。`source-manifest.json` 只记录仓库固定资产的来源，不证明另选权重的来源；加速模型的导出清单保留源权重内的许可、署名和 source/IR 哈希，使用其他权重时须保留其真实来源与适用许可。`THIRD_PARTY.md` 保留仓库资产的许可说明。CPU 与 GPU/NPU 安装均在 `share/yanliu/licenses/chinese-ime-lm/LICENSE` 保留完整 Apache-2.0 许可，在 `share/yanliu/licenses/chinese-ime-training/NOTICE` 保留完整上游署名声明。

安装先在临时目录准备资产，再提交完整文件组。初次安装拒绝覆盖已有同名文件，写入失败撤销本次新建文件。

升级前先正常退出砚流服务，等服务完成学习快照写入并退出，再执行：

```sh
./packaging/install-local.sh --upgrade
```

升级先校验当前 `installation.json` 及全部受管理文件的 SHA-256，并检查启动入口、组件、环境文件和配置仍指向这份安装。启动入口支持当前符号链接，以及早期安装实际使用、以 `#!/bin/sh`、`set -eu` 和一条 `exec` 转发到受管理服务的精确 755 脚本；不执行脚本，也不接受其他目标或额外命令。安装清单支持当前工具的 `model` 记录，以及本机早期安装实际使用的 `selected_model` / `model_provenance` 记录；其他形状明确拒绝，不猜测迁移。已有配置、启动入口、组件 XML 和环境文件逐字保留。配置的设备需与 `YANLIU_DEVICE` 一致，`data_dir`、显式 `model` 和 `runtime_dir` 需指向受管理目录；外部模型或数据目录应由用户单独维护。

新资产准备好后，旧资产移到同盘的 `share/.yanliu-previous`，新资产再移入 `share/yanliu`。当前目录中的非托管文件、目录和符号链接随安装移动到新的当前目录，不读取正文；若与新资产同名则拒绝升级。失败时恢复当前和上一版资产；如果恢复本身也失败，保留整个暂存目录，并在原错误中给出实际路径供人工恢复，不自动清除其中的资产或自定义文件。连续升级只保留上一版；更老备份须通过哈希校验且没有非托管内容，才会在成功后清除。

回到上一版资产使用：

```sh
./packaging/install-local.sh --rollback
```

回退交换当前和上一版资产，可再次执行以恢复。非托管内容仍留在当前目录，配置和系统学习不参与复制、备份或回退；若用户另往备份目录放了文件，须先移出这些文件才允许交换。加速配置只有在上一版包含匹配源权重的 IR 和所选设备运行库时才允许回退；回退不需要重新编译或设置源模型目录。回退不修改设备选择。

升级和回退均不查找、停止或启动桌面进程，也不改变输入源。操作完成后应重新加载砚流 IBus 服务；仍运行的旧服务可能持有旧模型和二进制。首次使用或组件发现有变化时，注销并重新登录。自定义前缀和配置根目录需要一直传入同样的 `YANLIU_PREFIX`、`XDG_CONFIG_HOME`。

卸载使用：

```sh
./packaging/uninstall-local.sh
```

自定义前缀时，卸载传入同一个 `YANLIU_PREFIX`。卸载保留 `yanliu/config.json` 和个人学习，删除整个当前资产目录、启动链接及环境文件；资产目录中的自定义文件须先移出。上一版备份保留在 `share/.yanliu-previous`，确认不再需要后可手工删除。卸载不更改 GNOME 输入源或其他输入法。

默认配置位于 `~/.config/yanliu/config.json`，绝对 `XDG_CONFIG_HOME` 可替换配置根目录。新配置只有 `data_dir`；服务支持的完整字段见[配置参考](../docs/USER-GUIDE.md#配置)。已有配置逐字保留，新配置与安装在同一事务内创建；安装读取 JSON 中的设备选择以检查构建是否匹配，服务启动时严格检查完整配置。配置属于用户状态，不进入安装哈希。未知字段与类型错误由服务明确拒绝，不自动迁移旧配置。下载过的模型/runtime 与个人数据均不自动删除或改写。

组件命令只携带 `--ibus --config` 与配置绝对路径，配置路径含空格或中文时正确引用；无设置程序或 `<setup>`。IBus 不扫描用户组件目录，因此安装写入 `~/.config/environment.d/60-yanliu.conf`，将 `IBUS_COMPONENT_PATH` 指向本项目组件目录及安装时已有的系统组件目录。注销重新登录后，在 GNOME 设置 → 键盘 → 输入源中添加“砚流”；安装不会自动选择它。

未提交拼音、光标与候选通过 IBus 交给 GNOME Shell；应用只接收隐藏的空预编辑，确认后才提交文本。逐键先发布静态候选，共享模型线程随后更新排序；导航、焦点、重置和敏感模式使迟到结果失效。

私有协议验证不连接用户桌面的 IBus：

```sh
cargo build --locked --release --features ibus --bins
/usr/bin/python3 scripts/test_ibus.py target/release/yanliu-ibus
/usr/bin/python3 tests/ibus_packaging.py
```

系统需有 `ibus-daemon` 和 Python GI/IBus typelib。协议脚本使用临时目录与私有地址，结束时清理自己的服务和 daemon。单入口覆盖引擎、焦点、输入操作、学习错误、用户词表和语境学习；组件部分使用正式 XML，验证发现和自动启动真实二进制，不预启动服务。安装 fixture 验证事务、配置、文件布局和卸载，不代替真实二进制验证。`scripts/test_ibus.py /absolute/prefix/bin/yanliu-ibus --installed /absolute/prefix` 可验证临时前缀的实际安装。

IBus 上游 XML 将 `FocusId` / `ActiveSurroundingText` 标为 `(b)`，但实际 C getter 与 engineproxy 读取布尔 `b`；服务遵守真实 getter 契约。`ContentType` 可写属性为 `(uu)`，密码/PIN/PRIVATE/HIDDEN_TEXT 透传，隐藏输入面板，不采集上下文或学习。

`tests/ibus_latency.py` 可测私有总线往返、静态候选信号和改变排序的信号时延，最终菜单逐项核对同步 CLI。它验证异步一致性，不证明中文质量、Shell 绘制或键到屏幕延迟。

可选 GPU / NPU 排序使用同一份默认 4M 权重。先建立一次性导出环境，准备独立模型目录和原生运行库；这些步骤只在构建机运行，系统服务不启动 Python：

```sh
python3 -m venv .cache/yanliu-export
.cache/yanliu-export/bin/python -m pip install -r scripts/accelerator-requirements.txt
.cache/yanliu-export/bin/python scripts/export_ranker.py \
  data/models/sentence-model.safetensors --output-dir .cache/yanliu-ranker-4m
cp data/models/sentence-model.safetensors .cache/yanliu-ranker-4m/sentence-model.safetensors
python3 scripts/prepare_accelerator_runtime.py .cache/yanliu-openvino
cargo build --locked --release --features ibus,accelerator --bins
target/release/yanliu --model .cache/yanliu-ranker-4m/sentence-model.safetensors \
  --device npu --runtime-dir .cache/yanliu-openvino decode xian --context 我去
```

两个准备脚本拒绝覆盖既有输出。运行库脚本默认从官方固定 URL 冷获取并验 archive SHA，也可用 `--archive /path/archive.tar.gz` 复用相同文件；模型导出清单绑定 source、XML、BIN、config、vocab 与许可。源权重不被修改。独立模型目录使已有 CPU 资产与相邻研究 IR 无需改写。导出不训练模型。

在本机已具备 Intel GPU/NPU 驱动、设备访问权限且上述显式调用成功后，可以安装：

```sh
YANLIU_DEVICE=npu \
YANLIU_MODEL_DIR="$PWD/.cache/yanliu-ranker-4m" \
YANLIU_ACCELERATOR_RUNTIME_DIR="$PWD/.cache/yanliu-openvino" \
./packaging/install-local.sh
```

改为 `gpu` 使用 GPU FP32；`cpu` 为默认，可用 `YANLIU_MODEL_DIR` 选择源权重，但不接受 `YANLIU_ACCELERATOR_RUNTIME_DIR`。GPU/NPU 的显式模型目录必须同时包含 `sentence-model.safetensors`、`.xml`、`.bin` 和 `.openvino.json`；安装提交前核对源权重及两个 IR 文件与导出清单的哈希。安装只带选中设备需要的库、IR 与许可，省略 GenAI、tokenizers 和 CPU 插件；不自动安装系统驱动。全部产物哈希随安装保留。初次新建的加速配置包含 `device` 和安装后 `runtime_dir`，旧配置仍逐字保留；若旧配置设备与 `YANLIU_DEVICE` 冲突，先明确编辑配置，再执行安装。服务不会自动猜测设备或在失败时改用 CPU。NPU 使用设备默认精度，近似并列时排序可能与 CPU 不同。
