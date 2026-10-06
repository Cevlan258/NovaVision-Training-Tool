"""从真实数据只读复制隔离测试集，避免性能验证写入用户数据。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("测试目录已存在，拒绝覆盖")
    destination.mkdir(parents=True)
    (destination / ".novavision-performance-fixture").write_text(
        str(source), encoding="utf-8"
    )
    copied = 0
    copied_bytes = 0
    records = []
    for image in sorted((source / "images").rglob("*")):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        relative = image.relative_to(source / "images")
        target = destination / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        label = source / "labels" / relative.with_suffix(".txt")
        label_hash = None
        if label.is_file():
            target_label = destination / "labels" / relative.with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, target_label)
            label_hash = hashlib.sha256(label.read_bytes()).hexdigest()
        records.append({"image": relative.as_posix(), "label_sha256": label_hash})
        copied += 1
        copied_bytes += image.stat().st_size
        if copied >= args.limit:
            break
    for name in ("classes.txt",):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    # 配置只引用测试副本，训练及验证清单由原程序在隔离目录生成。
    (destination / "data.yaml").write_text(
        "path: " + destination.as_posix() + "\ntrain: images/train\nval: images/val\n"
        "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n",
        encoding="utf-8",
    )
    report = {"source": str(source), "destination": str(destination), "images": copied,
              "image_bytes": copied_bytes, "records": records}
    (destination / "fixture-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
