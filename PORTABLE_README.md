# NovaVision Windows x64 便携训练工具

## 解压后启动

将完整包解压到一个可写目录，双击 `Start_NovaVision.cmd`。启动入口会按当前解压位置配置 Python 和桥接路径，并使用包内 .NET 9 运行库；无需安装 Python、.NET、开发 SDK，也无需手动校验文件。

请从这个启动入口打开程序，以便在移动目录后更新配置。不直接双击 `App_NovaVision/NovaVision.exe`。

系统要求：Windows 10/11 x64。GPU 训练需要支持 CUDA 12.8 的 NVIDIA 显卡及兼容驱动；没有合适显卡时可选择 CPU。包内不会安装或修改显卡驱动。

## 包内内容

- NovaVision 主程序、现有性能扩展及 Windows x64 原生依赖。
- Python 3.12、现有第三方包、PyTorch 2.7.1+cu128 和 Ultralytics 8.4.152。
- .NET 9.0.20 的 Core 与 WindowsDesktop 运行库，不含开发 SDK。
- 桥接脚本、一份 YOLOv5 兼容源码、现有应用预训练模型与 SAM 资源。
- 操作手册、图标和相关第三方许可证。

未打包本机授权文件、本机路径配置、日志、历史训练结果、用户数据集、开发工具链、程序集备份及临时修改工具。主程序使用本地现有版本，打包过程不另改授权逻辑。

训练时自行选择数据集、输出位置及模型。包内已有模型可离线选择；未附带的模型、特殊导出后端和额外资源仍可能需要单独准备。

## GitHub Releases 下载方式

完整 `NovaVision-Windows-x64.zip` 用于本地整体交付。由于 GitHub Releases 单附件必须小于 2 GiB，发布时上传 `NovaVision-Windows-x64-partNN.zip` 全部文件和本说明。

用户下载全部 `partNN.zip`，分别解压到同一个目录。每个 ZIP 都包含相同的顶层 `NovaVision/` 路径，各包文件互不重复；合并目录即可。它们是独立 ZIP，无需拼接，也无需额外解压软件。全部解压后双击 `NovaVision/Start_NovaVision.cmd`。

不要把这些数 GiB 的二进制附件作为普通 Git 源码文件提交。

GitHub 官方说明：https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
