<p align="center">
  <img src="docs/assets/readme-hero.svg" alt="砚流 Yanliu — 离线落字，自在成文。面向 GNOME / Wayland 的中文拼音输入法。右侧为输入示意。" width="100%">
</p>

<h1 align="center">砚流 · Yanliu</h1>

<p align="center">
  用 Rust 写的离线中文拼音输入法<br>
  本地词库召回 · 语境候选排序 · 个人选词学习
</p>

<p align="center">
  <a href="https://github.com/NEX-S/yanliu/actions/workflows/ci.yml"><img src="https://github.com/NEX-S/yanliu/actions/workflows/ci.yml/badge.svg" alt="CI 状态"></a>
  <a href="https://github.com/NEX-S/yanliu/releases/tag/v0.1.0"><img src="https://img.shields.io/badge/preview-v0.1.0-3e7665?style=flat-square" alt="技术预览 v0.1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-3e7665?style=flat-square" alt="GPL-3.0-or-later"></a>
  <img src="https://img.shields.io/badge/desktop-GNOME%20%2F%20Wayland-536d78?style=flat-square" alt="GNOME / Wayland">
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/USER-GUIDE.md">使用手册</a> ·
  <a href="packaging/README.md">安装与卸载</a> ·
  <a href="docs/DEVELOPMENT.md">开发指南</a> ·
  <a href="https://github.com/NEX-S/yanliu/releases/tag/v0.1.0">下载模型</a>
</p>

---

砚流通过 **IBus** 接入桌面，由 **GNOME Shell** 显示候选面板。词库负责切分与组合，本地字符语言模型结合应用提供的前文，从真实召回结果中提升候选；明确的选词会积累为本地偏好。

首次获取依赖与资产需要网络，**日常输入完全离线**。默认走轻量 CPU 路径；需要时可显式启用 OpenVINO，在 GPU / NPU 上运行同源模型。

> [!IMPORTANT]
> 当前为 **v0.1.0 技术预览**，候选仍可能误排。随附合成数据用于开发与回归，不代表自然用户准确率。横幅中的候选面板为输入示意。

## 为中文输入而做

- **先看到候选，再结合前文排序。** 逐键发布词库候选，本地模型随后更新；排序只提升真实候选，保留其原始身份与消费范围。
- **把输入留在自己的设备上。** 密码与敏感字段直接透传，不采集上下文或学习；默认偏好只保存明确选中的拼音、文本与次数。
- **输入可以接着改。** 支持拼音光标编辑、部分选词、音节分隔，以及相邻换位、邻键替换和漏键恢复。
- **词表与学习由你掌握。** 可加载自备 Rime 子集词表；学习可关闭、查看、清除，语境学习需显式开启。
- **桌面使用，终端也能验证。** CLI 与 IBus 共用 Session 输入处理路径，可直接检查候选和回放公开案例。

## 快速开始

需要 **Rustup、系统 C 链接器、Python 3.11+ 和 IBus**。仓库固定 Rust 1.98.0；构建应在实际使用的机器上进行。

<details>
<summary>安装 IBus 与协议测试依赖</summary>

Debian / Ubuntu：

```sh
sudo apt install ibus python3-gi gir1.2-ibus-1.0
```

Fedora：

```sh
sudo dnf install ibus ibus-libs python3-gobject-base
```

Python GI 与 IBus typelib 用于协议测试；开发脚本使用虚拟环境时，Debian / Ubuntu 另需 `python3-venv`。本机验证环境为 Fedora 44 / GNOME Wayland，CI 使用 Ubuntu 24.04。

</details>

**1. 获取源码与固定资产**

```sh
git clone https://github.com/NEX-S/yanliu.git
cd yanliu
python3 scripts/fetch_assets.py
```

下载时校验 SHA-256；已有资产可用 `python3 scripts/fetch_assets.py --verify-only` 验证。

**2. 安装到当前用户**

```sh
./packaging/install-local.sh
```

脚本会构建 CPU release 服务，并安装到 `~/.local`。完成后**注销并重新登录**，在 **GNOME 设置 → 键盘 → 输入源**中添加“砚流”。安装不会自动切换输入源。

**3. 开始输入**

键入 `nihao`，按空格选择“你好”。`Enter` 提交拼音原文，`Esc` 取消，`Ctrl+Space` 切换中英文模式。

升级或重装前先运行 `./packaging/uninstall-local.sh`；卸载保留配置和个人学习。自定义前缀、GPU / NPU 与完整安装过程见[安装说明](packaging/README.md)。

### 先在终端试一试

无需安装到桌面即可检查候选：

```sh
cargo build --locked --release --bin yanliu
target/release/yanliu decode nihao

# 加载本地模型，结合前文排序
target/release/yanliu --model data/models/sentence-model.safetensors \
  decode xian --context 我去
```

CLI 不带 `--model` 时只使用词库。`decode` 与 `evaluate` 不读写系统学习状态；批量案例与训练工具见[开发指南](docs/DEVELOPMENT.md)。

## 日常操作

- **选词：** 空格确认高亮候选，数字 `1–9` 选择当前页候选；部分选择保留剩余拼音。
- **候选导航：** 上下方向键或 `Ctrl+H/L` 移动，`PageUp/PageDown` 或 `Ctrl+K/J` 翻页。
- **编辑拼音：** 左右方向键移动光标，`Home/End` 跳到首尾；`Enter` 提交原文，`Esc` 取消。
- **英文与标点：** `Ctrl+Space` 切换英文透传；中文模式转换常用中文标点，单引号用于分隔音节。

拼音显示在候选面板中，`▏` 标出编辑位置，确认前不改应用正文。完整按键行为、标点规则与敏感字段处理见[输入操作](docs/USER-GUIDE.md#输入操作)。

## 选择模型

**默认 4M · 开箱路径**

`fetch_assets.py` 获取固定的公共原版 4M 字符模型，默认 CPU 安装直接使用它。

**可选 25M · 公开训练版**

[v0.1.0 发布附件](https://github.com/NEX-S/yanliu/releases/tag/v0.1.0) 提供 `yanliu-25m-v1.tar.gz` 与 `SHA256SUMS`，包含源权重和同源 OpenVINO IR。[模型卡](data/model-cards/yanliu-25m-v1/README.md)记录来源、哈希和评测限制，[训练配方](data/training/yanliu-25m-v1/README.md)可供复现。

解包后，可在首次 CPU 安装时指定：

```sh
YANLIU_MODEL_DIR=/absolute/path/to/yanliu-25m-v1 \
  ./packaging/install-local.sh
```

**可选 GPU / NPU · 显式启用**

需要 `accelerator` 构建、同源 IR、单独的 OpenVINO 原生运行库及可用硬件与驱动。运行库与驱动不包含在模型包中；设备失败会报出原因，不自动换回 CPU。GPU 使用 FP32，NPU 在近似并列时可能与 CPU 排序不同。准备与安装步骤见[设备配置](packaging/README.md)。

## 让它适合你的输入

安装后的配置位于 `~/.config/yanliu/config.json`，修改后需重启服务或重新登录。

- [配置参考](docs/USER-GUIDE.md#配置)：模型路径、设备、用户词表与学习开关。
- [本地学习](docs/USER-GUIDE.md#学习)：偏好保存、可选语境学习与数据清除。语境参数也可能记住私人信息，请按私人数据管理。
- [用户词表](docs/USER-GUIDE.md#用户词表)：使用你维护的 UTF-8 Rime 子集文件。

## 开发与贡献

项目保持单个 Rust 包：**decoder / 词库 → Session → model → IBus / CLI**。候选由词库召回，Session 管理输入和异步结果的有效性，模型负责排序，IBus 与 CLI 分别承担桌面协议和离线诊断。

从[开发指南](docs/DEVELOPMENT.md)开始，那里包含候选机制、CLI 评估、公开合成数据生成、训练与验证命令。开发工具可调用 deepseek-flash 生成公开合成案例；输入法运行时不接入远端模型。

欢迎在 [Issues](https://github.com/NEX-S/yanliu/issues) 提交可复现的问题。候选问题请附**最小公开输入、前文、期望候选与实际排名**，也可提供 `evaluate` 可读取的 JSONL。请勿提交私人应用上下文、词表、学习快照、密钥或本地对话记录。Agent 开发约束见 [AGENTS.md](AGENTS.md)。

## 许可与致谢

主项目采用 [GPL-3.0-or-later](LICENSE)。砚流使用的词库、模型和 vendor 保留各自许可与署名；固定来源和 SHA-256 见[资产清单](data/manifest.json)，完整说明见[第三方来源与许可](data/THIRD_PARTY.md)。

感谢开放词库、语言模型和输入法基础设施的维护者，让中文输入可以在本地继续生长。
