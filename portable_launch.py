"""按解压位置配置并启动便携版 NovaVision，不增加校验或安装步骤。"""

import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    app = root / "App_NovaVision"
    settings_path = app / "settings.json"
    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    settings.update({
        "PythonExecutablePath": str(root / "python.exe"),
        "BridgeScriptsDirectory": str(app / "ViTooAI_Bridge"),
    })
    settings_path.write_text(json.dumps(settings, ensure_ascii=True, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["DOTNET_ROOT"] = str(root / "dotnet")
    env["DOTNET_ROOT_X64"] = str(root / "dotnet")
    env["DOTNET_MULTILEVEL_LOOKUP"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PATH"] = os.pathsep.join((str(root), str(root / "DLLs"), str(root / "dotnet"), env.get("PATH", "")))
    subprocess.Popen([str(app / "NovaVision.exe")], cwd=app, env=env)


if __name__ == "__main__":
    main()
