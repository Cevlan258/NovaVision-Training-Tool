<div align="center">
  <img src="novavision_icon.png" alt="NovaVision" width="120" />
  <h1>NovaVision</h1>
  <p><strong>Windows 便携视觉训练工具</strong></p>
  <p>从数据标注、图像增强到模型训练、导出与推理，连接完整视觉工作流。</p>
  <p>
    <img alt="Windows x64" src="https://img.shields.io/badge/Windows-x64-0078D4?style=flat-square" />
    <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square" />
    <img alt="PyTorch 2.7.1" src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=flat-square" />
    <img alt="CUDA 12.8" src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square" />
    <img alt=".NET 9" src="https://img.shields.io/badge/.NET-9-512BD4?style=flat-square" />
  </p>
  <p>
    <a href="https://github.com/Cevlan258/NovaVision-Training-Tool/releases/latest"><strong>下载便携版</strong></a> ·
    <a href="#三步开始使用">快速开始</a> ·
    <a href="NovaVision_用户操作手册.md">完整操作手册</a> ·
    <a href="https://github.com/Cevlan258/NovaVision-Training-Tool/issues">问题反馈</a>
  </p>
</div>


---

NovaVision 将数据标注、图像增强、YOLO 训练、模型导出与推理集中在 Windows 桌面界面中。便携版附带主程序和运行环境，免激活，解压后即可通过启动入口使用，无需安装 Python 或 .NET。

## 主要功能

| 功能 | 说明 |
| :--- | :--- |
| 🏷️ 数据标注 | 检测框、多边形、旋转框、关键点及图像分类标注 |
| ✨ 图像增强 | 旋转、翻转、光照调整等离线增强 |
| 🎯 模型训练 | 配置数据集、模型和训练参数，支持断点续训入口 |
| 📦 导出与推理 | 模型评估、ONNX 等格式导出、图片与视频推理 |

覆盖 **Detect / Segment / OBB / Pose / Classify** 五种任务，具体操作见 [用户手册](NovaVision_用户操作手册.md)。

## 三步开始使用

1. **下载全部六个 ZIP。** 从 [最新发行版](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/latest) 下载下表全部附件。
2. **解压到同一个目录。** 六个独立 ZIP 会合并为 `NovaVision/` 文件夹，无需拼接或额外解压软件。
3. **双击启动。** 运行 `NovaVision/Start_NovaVision.cmd`，选择数据集、模型、设备与输出位置。

| 附件 | 大小 | 下载 |
| :--- | ---: | :---: |
| `part01.zip` | 1,604 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part01.zip) |
| `part02.zip` | 1,324 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part02.zip) |
| `part03.zip` | 562 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part03.zip) |
| `part04.zip` | 1,177 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part04.zip) |
| `part05.zip` | 969 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part05.zip) |
| `part06.zip` | 923 MiB | [下载](https://github.com/Cevlan258/NovaVision-Training-Tool/releases/download/v2026.10.07/NovaVision-Windows-x64-part06.zip) |

总下载量约 **6.41 GiB**，解压后约 **9.12 GiB**。请下载六个便携版附件；GitHub 的 **Source code (zip)** 仅包含源码。

## 系统要求

- **系统：** Windows 10 / 11，x64。
- **GPU：** GPU 训练需要支持 CUDA 12.8 的 NVIDIA 显卡与兼容驱动；也可在支持的任务入口选择 CPU。
- **空间：** 除程序外，为数据集和训练结果预留磁盘空间。

移动程序目录后，仍通过 `Start_NovaVision.cmd` 启动，路径会自动更新。数据集需自行准备。

## 使用帮助

[完整操作手册](NovaVision_用户操作手册.md) · [便携版说明](PORTABLE_README.md) · [问题反馈](https://github.com/Cevlan258/NovaVision-Training-Tool/issues)

## 许可证

YOLOv5 遵循 [AGPL-3.0](App_NovaVision/yolov5/LICENSE)，其他依赖、模型及资源遵循各自许可条件。本仓库未为全部源码和桌面二进制统一授予新的开源许可证。

---

<div align="center">
  <sub>NovaVision · 数据标注 → 图像增强 → 模型训练 → 导出与推理</sub>
</div>
