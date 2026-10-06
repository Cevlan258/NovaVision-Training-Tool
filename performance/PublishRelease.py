"""生成带哈希清单的本地交付目录，不改写运行中的应用。"""

import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    app = base.parent / "App_NovaVision"
    release = base / "release"
    release.mkdir(exist_ok=True)
    shutil.copy2(base / "staging" / "NovaVision.dll", release / "NovaVision.dll")
    shutil.copy2(base / "build" / "helper" / "NovaVision.Performance.dll", release / "NovaVision.Performance.dll")
    deps = json.loads((app / "NovaVision.deps.json").read_text(encoding="utf-8-sig"))
    for target in deps["targets"].values():
        root = next(key for key in target if key.startswith("NovaVision/"))
        target[root].setdefault("dependencies", {})["NovaVision.Performance"] = "1.0.0"
        target["NovaVision.Performance/1.0.0"] = {
            "runtime": {"NovaVision.Performance.dll": {"assemblyVersion": "1.0.0.0", "fileVersion": "1.0.0.0"}}
        }
    deps["libraries"]["NovaVision.Performance/1.0.0"] = {"type": "project", "serviceable": False, "sha512": ""}
    (release / "NovaVision.deps.json").write_text(json.dumps(deps, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "original_dll_sha256": digest(base / "backups" / "assembly-original" / "NovaVision.dll"),
        "original_deps_sha256": digest(base / "backups" / "assembly-original" / "NovaVision.deps.json"),
        "files": {path.name: digest(path) for path in release.iterdir() if path.name != "manifest.json" and path.is_file()},
    }
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
