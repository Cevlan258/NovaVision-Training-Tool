"""验证真实兼容入口可以解析 CLI，禁止启动训练或模型推理。"""

import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    results = []
    for script in sorted((root / "ViTooAI_Bridge").glob("bridge_*.py")):
        try:
            process = subprocess.run(
                [str(root / "python.exe"), "-B", str(script), "--help"],
                cwd=root,
                env={**os.environ, "PYTHONUTF8": "1"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
            )
            result = {"script": script.name, "exit_code": process.returncode,
                      "passed": process.returncode == 0 and "usage:" in process.stdout.lower(),
                      "output": (process.stdout + process.stderr)[-2000:]}
        except subprocess.TimeoutExpired:
            result = {"script": script.name, "passed": False, "timeout": True}
        results.append(result)
        print(script.name, "PASS" if result["passed"] else "FAIL", flush=True)
    (root / "performance" / "reports" / "bridge-cli.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
