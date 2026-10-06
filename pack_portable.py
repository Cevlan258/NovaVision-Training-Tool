"""整理现有运行环境并生成完整 ZIP 与 GitHub Releases 独立 ZIP 附件。"""

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "portable-release"
PACKAGE = OUTPUT / "NovaVision"
PART_LIMIT = 1792 * 1024 * 1024


def include(path):
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if "__pycache__" in parts or path.suffix.lower() in {".pyc", ".pyo", ".bak", ".log", ".lic", ".lib"}:
        return False
    if parts[0] == "App_NovaVision":
        if len(parts) > 1 and parts[1] in {"backups", "docs", "InferenceOutputs", "log", "logs", "Sample"}:
            return False
        if len(parts) == 2 and path.name in {"settings.json", "last_dataset.txt"}:
            return False
        if len(parts) > 2 and parts[1] == "runtimes" and parts[2] not in {"win", "win-x64"}:
            return False
        if path.suffix.lower() == ".pdf":
            return False
    if parts[:2] == ("Lib", "test"):
        return False
    if parts[:4] == ("Lib", "site-packages", "torch", "include"):
        return False
    return True


def write_zip(path, paths):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source in paths:
            archive.write(source, "NovaVision/" + source.relative_to(PACKAGE).as_posix())


def main():
    # 不覆盖既有交付包，也不改写原应用目录。
    if OUTPUT.exists():
        raise SystemExit("portable-release 已存在，请另行保留或移走旧交付包后重新打包。")
    files = []
    for directory in ("App_NovaVision", "DLLs", "Lib", "Scripts", "share", "tcl", "ViTooAI_Bridge"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and include(path))
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "Arial.ttf", "Arial.Unicode.ttf", "LICENSE.txt", "novavision.ico", "novavision_icon.png", "NovaVision_用户操作手册.md", "yolo26n.pt", "yolov8n.pt", "portable_launch.py", "Start_NovaVision.cmd", "PORTABLE_README.md"):
        files.append(ROOT / name)
    sdk = ROOT / "performance" / "toolchain" / "dotnet"
    runtime_files = [sdk / "dotnet.exe", sdk / "LICENSE.txt", sdk / "ThirdPartyNotices.txt"]
    for directory in ("host/fxr/9.0.20", "shared/Microsoft.NETCore.App/9.0.20", "shared/Microsoft.WindowsDesktop.App/9.0.20"):
        runtime_files.extend(path for path in (sdk / directory).rglob("*") if path.is_file())
    assert all(path.is_file() for path in files + runtime_files)
    PACKAGE.mkdir(parents=True)
    print(f"Copying {len(files)} application/Python files and {len(runtime_files)} .NET runtime files", flush=True)
    for source in files:
        target = PACKAGE / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in runtime_files:
        target = PACKAGE / "dotnet" / source.relative_to(sdk)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # CMD 使用 ASCII 和 CRLF，兼容默认 Windows 命令解释器。
    command = (PACKAGE / "Start_NovaVision.cmd").read_text(encoding="utf-8")
    (PACKAGE / "Start_NovaVision.cmd").write_bytes(command.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
    paths = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    print(f"Staged {len(paths)} files, {sum(p.stat().st_size for p in paths)} bytes", flush=True)
    write_zip(OUTPUT / "NovaVision-Windows-x64.zip", paths)
    print("Full ZIP ready", flush=True)
    groups = [[]]
    size = 0
    for path in paths:
        length = path.stat().st_size
        if length >= PART_LIMIT:
            raise RuntimeError(f"单文件过大: {path}")
        if size + length > PART_LIMIT:
            groups.append([])
            size = 0
        groups[-1].append(path)
        size += length
    for number, group in enumerate(groups, 1):
        output = OUTPUT / f"NovaVision-Windows-x64-part{number:02d}.zip"
        write_zip(output, group)
        assert output.stat().st_size < 2 * 1024**3
        print(f"Release asset {number}/{len(groups)} ready: {output.stat().st_size} bytes", flush=True)
    shutil.copy2(ROOT / "PORTABLE_README.md", OUTPUT / "PORTABLE_README.md")
    summary = {"files": len(paths), "unpacked_bytes": sum(p.stat().st_size for p in paths), "archives": {p.name: p.stat().st_size for p in OUTPUT.glob("*.zip")}}
    (OUTPUT / "package-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
