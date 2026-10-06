from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "191A01EA651A721CEA708151A4F44617B6FE8C01F31D23BE2E82E5F2A207BE7F"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 NovaVision 候选补丁的 token、资源和方法语义 allowlist。")
    parser.add_argument("candidate", type=Path, help="patcher 生成的候选 NovaVision.dll")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "App_NovaVision" / "NovaVision.dll",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "patch-audit.json",
    )
    parser.add_argument(
        "--dotnet",
        type=Path,
        default=Path(__file__).resolve().parent / "toolchain" / "dotnet" / "dotnet.exe",
    )
    parser.add_argument(
        "--patcher",
        type=Path,
        default=Path(__file__).resolve().parent / "bin" / "Debug" / "net9.0" / "Patcher.dll",
    )
    args = parser.parse_args()

    original = args.original.resolve()
    candidate = args.candidate.resolve()
    report = args.report.resolve()
    dotnet = args.dotnet.resolve()
    patcher = args.patcher.resolve()

    for path in (original, candidate, dotnet, patcher):
        if not path.is_file():
            parser.error(f"文件不存在: {path}")
    if original == candidate:
        parser.error("候选路径不能与源 DLL 相同")
    if sha256(original) != EXPECTED_SOURCE_SHA256:
        parser.error("源 DLL SHA256 与已审计基线不匹配")

    report.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(dotnet), str(patcher), "verify", str(original), str(candidate), str(report)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    audit = json.loads(report.read_text(encoding="utf-8-sig"))
    if audit.get("Status") != "PASS":
        print("审计报告没有 PASS 状态。", file=sys.stderr)
        return 1
    if audit.get("OriginalSha256", "").upper() != EXPECTED_SOURCE_SHA256:
        print("报告中的源 DLL SHA256 与基线不匹配。", file=sys.stderr)
        return 1
    if not audit.get("ResourcesBytePreserved") or not audit.get("ExistingMetadataTokensPreserved"):
        print("报告未确认资源字节和既有 metadata token 保持不变。", file=sys.stderr)
        return 1
    if sha256(original) != EXPECTED_SOURCE_SHA256:
        print("验证过程中源 DLL 内容发生变化。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
