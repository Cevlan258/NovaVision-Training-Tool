# 发行维护

源码仓库和便携发行附件承担不同用途。下载即用的版本放在 GitHub Releases；Git 仓库保留源码、脚本和说明。

## 本地准备

`pack_portable.py` 依赖完整本地应用和 Python 环境，以及 `performance/toolchain/dotnet` 下的 .NET 9.0.20 运行文件。本机环境包括已有主程序、性能扩展、预训练模型及桥接实现；这些大体积运行文件不随源码仓库提交。

在准备完整的项目目录执行：

```powershell
python -B pack_portable.py
```

工具生成 `portable-release/NovaVision/`、完整 ZIP、六个独立 ZIP 附件及便携说明，拒绝覆盖已有 `portable-release/`。重新打包前应另行保留或移走旧交付包，不覆盖运行中的应用。

## GitHub Releases

上传全部 `NovaVision-Windows-x64-partNN.zip`，附 `PORTABLE_README.md`，并在发行说明中明确全部附件需要解压到同一目标目录。完整 ZIP 超过单附件限制，仅用于本地交付。

发行前核对包内主程序版本、附件完整性、合并覆盖范围、启动路径与运行依赖。维护者的发行验证不增加最终用户的校验步骤。版本更新时同步主页下载链接、附件大小和验证记录。

本次使用日期版本 `v2026.10.07`，表示这次便携发行，不代表上游桌面主程序的原始版本号。
