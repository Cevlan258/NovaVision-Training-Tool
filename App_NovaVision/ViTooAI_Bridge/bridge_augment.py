# -*- coding: utf-8 -*-
"""
NovaVision Studio - YOLO Dataset Offline Augmentation Bridge
Generates augmented images and accurately transforms/generates corresponding YOLO labels.
Supports Unicode / Chinese file paths on Windows.
"""
import os
import sys
import argparse
import random
import cv2
import numpy as np
import json
from pathlib import Path

def cv_imread(file_path):
    """Safely read images from Unicode/Chinese paths on Windows."""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def cv_imwrite(file_path, img):
    """Safely write images to Unicode/Chinese paths on Windows."""
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = '.jpg'
        ret, buf = cv2.imencode(ext, img)
        if ret:
            buf.tofile(file_path)
            return True
        return False
    except Exception:
        return False

def adjust_brightness_contrast(img, brightness_range=(-40, 40), contrast_range=(0.7, 1.3)):
    b = random.randint(brightness_range[0], brightness_range[1])
    c = random.uniform(contrast_range[0], contrast_range[1])
    return cv2.convertScaleAbs(img, alpha=c, beta=b)

def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = (h.astype(np.float32) * (1 + random.uniform(-h_gain, h_gain))) % 180
    s = np.clip(s.astype(np.float32) * (1 + random.uniform(-s_gain, s_gain)), 0, 255)
    v = np.clip(v.astype(np.float32) * (1 + random.uniform(-v_gain, v_gain)), 0, 255)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def add_fog(img, fog_coeff=0.3):
    fog = np.ones_like(img) * 255
    return cv2.addWeighted(img, 1 - fog_coeff, fog, fog_coeff, 0)

def add_shadow(img, shadow_intensity=0.4):
    h, w = img.shape[:2]
    x1, y1 = random.randint(0, w), 0
    x2, y2 = random.randint(0, w), h
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return cv2.addWeighted(img, 1, overlay, -shadow_intensity, 0)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def add_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def add_noise(img, amount=0.03):
    noise = np.random.randn(*img.shape) * (amount * 255)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_rain(img, rain_intensity=0.04):
    h, w = img.shape[:2]
    noise = np.random.rand(h, w) < rain_intensity
    rain_img = img.copy()
    rain_img[noise] = [200, 200, 200]
    return cv2.blur(rain_img, (3, 3))

def parse_line_tokens(line):
    clean = line.strip().lstrip('\ufeff')
    return [t.strip().lstrip('\ufeff') for t in clean.split() if t.strip()]

def load_yaml_config(input_dir):
    """Safely load data.yaml if present in input_dir or parent directories."""
    cur = os.path.abspath(input_dir)
    for _ in range(4):
        for name in ('data.yaml', 'data.yml'):
            candidate = os.path.join(cur, name)
            if os.path.isfile(candidate):
                try:
                    import yaml
                    with open(candidate, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            return data
                except Exception:
                    pass
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {}

def transform_point(x, y, op):
    if op == 'fliplr':
        return 1.0 - x, y
    elif op == 'flipud':
        return x, 1.0 - y
    elif op == 'rot90':
        return 1.0 - y, x
    elif op == 'rot180':
        return 1.0 - x, 1.0 - y
    elif op == 'rot270':
        return y, 1.0 - x
    return x, y

def transform_bbox(cx, cy, w, h, op):
    ncx, ncy = transform_point(cx, cy, op)
    if op in ('rot90', 'rot270'):
        nw, nh = h, w
    else:
        nw, nh = w, h
    return (
        max(0.0, min(1.0, ncx)),
        max(0.0, min(1.0, ncy)),
        max(0.0001, min(1.0, nw)),
        max(0.0001, min(1.0, nh))
    )

def detect_line_format(parts, kpt_shape=None):
    """
    Detects format of a YOLO annotation line.
    Returns (fmt_type, extra_info):
      'point': (px, py)
      'box': (cx, cy, w, h)
      'pose': (kpt_dim, num_kpts)
      'polygon': num_points
      'unknown': None
    """
    n = len(parts)
    if n == 3:
        return 'point', None
    if n == 5:
        return 'box', None

    # Check pose with kpt_shape if explicitly known from data.yaml
    if kpt_shape and isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 2:
        k_num = int(kpt_shape[0])
        k_dim = int(kpt_shape[1])
        if n == 5 + k_num * k_dim:
            if k_dim >= 3:
                all_vis = True
                for i in range(k_num):
                    v_tok = parts[5 + k_dim * i + 2]
                    try:
                        v_num = float(v_tok)
                        if v_num in (0.0, 1.0, 2.0) and ('.' not in v_tok or v_tok.endswith('.0')):
                            continue
                        all_vis = False
                        break
                    except ValueError:
                        all_vis = False
                        break
                if all_vis:
                    return 'pose', (k_dim, k_num)
            else:
                return 'pose', (k_dim, k_num)

    # Check pose with dim=3 (visibility 0, 1, 2)
    if n > 5 and (n - 5) % 3 == 0:
        k_num = (n - 5) // 3
        all_vis = True
        for i in range(k_num):
            v_tok = parts[5 + 3 * i + 2]
            try:
                v_num = float(v_tok)
                if v_num in (0.0, 1.0, 2.0) and ('.' not in v_tok or v_tok.endswith('.0')):
                    continue
                all_vis = False
                break
            except ValueError:
                all_vis = False
                break
        if all_vis:
            return 'pose', (3, k_num)

    # Check polygon / OBB: (n - 1) is even, n >= 7
    if n > 5 and (n - 1) % 2 == 0:
        return 'polygon', (n - 1) // 2

    return 'unknown', None

def transform_single_line(line, op, kpt_shape=None, flip_idx=None):
    clean = line.strip().lstrip('\ufeff')
    if not clean:
        return ""
    parts = parse_line_tokens(clean)
    if not parts:
        return clean

    fmt_type, extra = detect_line_format(parts, kpt_shape)

    try:
        cid = parts[0]

        if fmt_type == 'box':
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            ncx, ncy, nw, nh = transform_bbox(cx, cy, w, h, op)
            return f"{cid} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"

        elif fmt_type == 'point':
            px, py = float(parts[1]), float(parts[2])
            npx, npy = transform_point(px, py, op)
            npx = max(0.0, min(1.0, npx))
            npy = max(0.0, min(1.0, npy))
            return f"{cid} {npx:.6f} {npy:.6f}"

        elif fmt_type == 'pose':
            k_dim, k_num = extra
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            ncx, ncy, nw, nh = transform_bbox(cx, cy, w, h, op)

            kpts = []
            for i in range(k_num):
                offset = 5 + k_dim * i
                kx = float(parts[offset])
                ky = float(parts[offset + 1])
                v_val = None
                if k_dim >= 3:
                    v_val = int(round(float(parts[offset + 2])))

                # Unannotated keypoint preservation
                if kx == 0.0 and ky == 0.0 and (v_val is None or v_val == 0):
                    nkx, nky = 0.0, 0.0
                else:
                    nkx, nky = transform_point(kx, ky, op)
                    nkx = max(0.0, min(1.0, nkx))
                    nky = max(0.0, min(1.0, nky))

                if k_dim >= 3:
                    kpts.append((nkx, nky, v_val))
                else:
                    kpts.append((nkx, nky))

            # Apply flip_idx on fliplr if provided
            if op == 'fliplr' and flip_idx and len(flip_idx) == k_num:
                if max(flip_idx) < k_num and min(flip_idx) >= 0:
                    kpts = [kpts[flip_idx[i]] for i in range(k_num)]

            out_tokens = [str(cid), f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
            for kp in kpts:
                if k_dim >= 3:
                    out_tokens.extend([f"{kp[0]:.6f}", f"{kp[1]:.6f}", str(kp[2])])
                else:
                    out_tokens.extend([f"{kp[0]:.6f}", f"{kp[1]:.6f}"])

            return " ".join(out_tokens)

        elif fmt_type == 'polygon':
            num_pts = extra
            out_pts = []
            for i in range(num_pts):
                px = float(parts[1 + 2 * i])
                py = float(parts[1 + 2 * i + 1])
                npx, npy = transform_point(px, py, op)
                npx = max(0.0, min(1.0, npx))
                npy = max(0.0, min(1.0, npy))
                out_pts.append(f"{npx:.6f} {npy:.6f}")
            return f"{cid} " + " ".join(out_pts)

        else:
            return clean

    except Exception:
        return clean

def transform_labels(lines, op, kpt_shape=None, flip_idx=None):
    new_lines = []
    for line in lines:
        t_line = transform_single_line(line, op, kpt_shape, flip_idx)
        if t_line:
            new_lines.append(t_line)
    return new_lines

def transform_labels_fliplr(lines, kpt_shape=None, flip_idx=None):
    return transform_labels(lines, 'fliplr', kpt_shape, flip_idx)

def transform_labels_flipud(lines, kpt_shape=None, flip_idx=None):
    return transform_labels(lines, 'flipud', kpt_shape, flip_idx)

def transform_labels_rot90(lines, kpt_shape=None, flip_idx=None):
    return transform_labels(lines, 'rot90', kpt_shape, flip_idx)

def transform_labels_rot180(lines, kpt_shape=None, flip_idx=None):
    return transform_labels(lines, 'rot180', kpt_shape, flip_idx)

def transform_labels_rot270(lines, kpt_shape=None, flip_idx=None):
    return transform_labels(lines, 'rot270', kpt_shape, flip_idx)

def main():
    parser = argparse.ArgumentParser(description="NovaVision Dataset Augmentation Bridge")
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--augmentations', type=str, default="brightness,contrast,hsv,fliplr,shadow,sharpen", help="Comma separated augmentation list")
    parser.add_argument('--multiplier', type=int, default=2, help="Number of augmented variants per image")
    parser.add_argument('--keep_original', action='store_true', help="Copy original images to output")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    selected_augs = [a.strip().lower() for a in args.augmentations.split(",") if a.strip()]

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
    tasks = []
    seen_imgs = set()

    ignored_subdirs = {
        'images', 'labels', 'train', 'val', 'test', '.git', '.vs',
        'runs', 'output', 'outputs', 'inferenceoutputs', 'augmented_dataset', '__pycache__',
        'weights', 'models', 'bin', 'obj', '.idea', '.vscode'
    }

    def find_label_file(img_path, img_dir, root_dir):
        base = os.path.splitext(os.path.basename(img_path))[0]
        cand1 = os.path.join(img_dir, base + ".txt")
        if os.path.isfile(cand1):
            return cand1, img_dir
        cand2 = os.path.join(img_dir, "labels", base + ".txt")
        if os.path.isfile(cand2):
            return cand2, os.path.join(img_dir, "labels")
        rel = os.path.relpath(img_path, root_dir)
        parts = Path(rel).parts
        if len(parts) > 1 and parts[0].lower() == 'images':
            lbl_rel = os.path.join('labels', *parts[1:-1], base + ".txt")
            cand3 = os.path.join(root_dir, lbl_rel)
            if os.path.isfile(cand3):
                return cand3, os.path.dirname(cand3)
        sub_rel = os.path.relpath(img_dir, root_dir)
        if sub_rel != '.':
            cand4 = os.path.join(root_dir, "labels", sub_rel, base + ".txt")
            if os.path.isfile(cand4):
                return cand4, os.path.dirname(cand4)
        cand5 = os.path.join(root_dir, "labels", base + ".txt")
        if os.path.isfile(cand5):
            return cand5, os.path.join(root_dir, "labels")
        return None, img_dir

    def add_img_task(fpath, fdir):
        real_fpath = os.path.abspath(fpath)
        if real_fpath in seen_imgs:
            return
        seen_imgs.add(real_fpath)
        fname = os.path.basename(fpath)
        lbl_file, lbl_dir = find_label_file(fpath, fdir, input_dir)
        rel_img_dir = os.path.relpath(fdir, input_dir)
        dst_img = input_dir if rel_img_dir == '.' else os.path.join(output_dir, rel_img_dir)
        dst_lbl = None
        if lbl_file:
            rel_lbl_dir = os.path.relpath(lbl_dir, input_dir)
            dst_lbl = input_dir if rel_lbl_dir == '.' else os.path.join(output_dir, rel_lbl_dir)
        tasks.append({
            'src_img': fpath,
            'src_lbl': lbl_file,
            'dst_img_dir': dst_img,
            'dst_lbl_dir': dst_lbl,
            'img_name': fname
        })

    # 1. First scan root images directly (e.g. flat datasets like LS/)
    if os.path.isdir(input_dir):
        for f in os.listdir(input_dir):
            fpath = os.path.join(input_dir, f)
            if os.path.isfile(fpath) and f.lower().endswith(img_exts) and not f.startswith('.') and '_aug_' not in f and '__aug_' not in f:
                add_img_task(fpath, input_dir)

    # 2. Scan images/train and images/
    for img_sub in ('images/train', 'images'):
        cand_dir = os.path.join(input_dir, *img_sub.split('/'))
        if os.path.isdir(cand_dir):
            for f in os.listdir(cand_dir):
                fpath = os.path.join(cand_dir, f)
                if os.path.isfile(fpath) and f.lower().endswith(img_exts) and not f.startswith('.') and '_aug_' not in f and '__aug_' not in f:
                    add_img_task(fpath, cand_dir)

    # 3. Scan train/ subfolders (classification nested)
    train_dir = os.path.join(input_dir, "train")
    if os.path.isdir(train_dir):
        for sub in os.listdir(train_dir):
            sub_path = os.path.join(train_dir, sub)
            if os.path.isdir(sub_path) and sub.lower() not in ignored_subdirs and not sub.startswith('.'):
                for f in os.listdir(sub_path):
                    fpath = os.path.join(sub_path, f)
                    if os.path.isfile(fpath) and f.lower().endswith(img_exts) and not f.startswith('.') and '_aug_' not in f and '__aug_' not in f:
                        add_img_task(fpath, sub_path)

    # 4. Scan arbitrary class subdirectories in input_dir (only if they actually contain image files)
    if os.path.isdir(input_dir):
        for sub in os.listdir(input_dir):
            sub_path = os.path.join(input_dir, sub)
            if os.path.isdir(sub_path) and sub.lower() not in ignored_subdirs and not sub.startswith('.'):
                for f in os.listdir(sub_path):
                    fpath = os.path.join(sub_path, f)
                    if os.path.isfile(fpath) and f.lower().endswith(img_exts) and not f.startswith('.') and '_aug_' not in f and '__aug_' not in f:
                        add_img_task(fpath, sub_path)

    yaml_config = load_yaml_config(input_dir)
    kpt_shape = yaml_config.get('kpt_shape', None)
    flip_idx = yaml_config.get('flip_idx', None)

    total_imgs = len(tasks)
    print(f"__AUG_START__{json.dumps({'total_source': total_imgs, 'augs': selected_augs, 'multiplier': args.multiplier})}", flush=True)

    count_processed = 0
    total_generated = 0

    for idx, t in enumerate(tasks):
        img_path = t['src_img']
        img_name = t['img_name']
        base_name = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        out_img_dir = t['dst_img_dir']
        out_lbl_dir = t['dst_lbl_dir']
        os.makedirs(out_img_dir, exist_ok=True)
        if out_lbl_dir:
            os.makedirs(out_lbl_dir, exist_ok=True)

        orig_labels = []
        if t['src_lbl'] and os.path.exists(t['src_lbl']):
            try:
                with open(t['src_lbl'], 'r', encoding='utf-8-sig', errors='ignore') as f:
                    orig_labels = [line.strip().lstrip('\ufeff') for line in f if line.strip().lstrip('\ufeff')]
            except Exception:
                orig_labels = []

        img = cv_imread(img_path)
        if img is None:
            continue

        # Keep original if requested and output is a different folder
        if args.keep_original and os.path.abspath(out_img_dir) != os.path.abspath(os.path.dirname(img_path)):
            dst_img = os.path.join(out_img_dir, img_name)
            cv_imwrite(dst_img, img)
            if orig_labels and out_lbl_dir:
                dst_lbl = os.path.join(out_lbl_dir, base_name + ".txt")
                with open(dst_lbl, 'w', encoding='utf-8') as f:
                    f.write("\n".join(orig_labels) + "\n")
            total_generated += 1

        # Generate N variants
        for m in range(args.multiplier):
            aug_choice = random.choice(selected_augs) if selected_augs else 'bright_contrast'
            aug_img = img.copy()
            cur_labels = list(orig_labels)

            if aug_choice in ('bright_contrast', 'brightness', 'contrast'):
                aug_img = adjust_brightness_contrast(aug_img, brightness_range=(-50, 50), contrast_range=(0.8, 1.2))
            elif aug_choice == 'hsv':
                aug_img = augment_hsv(aug_img, h_gain=0.3, s_gain=0.4, v_gain=0.4)
            elif aug_choice in ('gray', 'grayscale'):
                aug_img = to_grayscale(aug_img)
            elif aug_choice == 'fog':
                aug_img = add_fog(aug_img, fog_coeff=random.uniform(0.15, 0.4))
            elif aug_choice == 'shadow':
                aug_img = add_shadow(aug_img, shadow_intensity=random.uniform(0.25, 0.5))
            elif aug_choice == 'sharpen':
                aug_img = sharpen(aug_img)
            elif aug_choice == 'blur':
                aug_img = add_blur(aug_img, ksize=random.choice([3, 5, 7]))
            elif aug_choice == 'noise':
                aug_img = add_noise(aug_img, amount=random.uniform(0.02, 0.05))
            elif aug_choice == 'rain':
                aug_img = add_rain(aug_img, rain_intensity=random.uniform(0.02, 0.06))
            elif aug_choice == 'fliplr':
                aug_img = cv2.flip(aug_img, 1) # Horizontal flip
                if cur_labels: cur_labels = transform_labels_fliplr(cur_labels, kpt_shape=kpt_shape, flip_idx=flip_idx)
            elif aug_choice == 'flipud':
                aug_img = cv2.flip(aug_img, 0) # Vertical flip
                if cur_labels: cur_labels = transform_labels_flipud(cur_labels, kpt_shape=kpt_shape, flip_idx=flip_idx)
            elif aug_choice == 'rot90':
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
                if cur_labels: cur_labels = transform_labels_rot90(cur_labels, kpt_shape=kpt_shape, flip_idx=flip_idx)
            elif aug_choice == 'rot180':
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
                if cur_labels: cur_labels = transform_labels_rot180(cur_labels, kpt_shape=kpt_shape, flip_idx=flip_idx)
            elif aug_choice == 'rot270':
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if cur_labels: cur_labels = transform_labels_rot270(cur_labels, kpt_shape=kpt_shape, flip_idx=flip_idx)

            out_name = f"{base_name}_aug_{aug_choice}_{m+1:02d}{ext}"
            out_img_path = os.path.join(out_img_dir, out_name)

            cv_imwrite(out_img_path, aug_img)
            if cur_labels and out_lbl_dir:
                out_lbl_path = os.path.join(out_lbl_dir, f"{base_name}_aug_{aug_choice}_{m+1:02d}.txt")
                with open(out_lbl_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(cur_labels) + "\n")

            total_generated += 1

        count_processed += 1
        progress_payload = {
            "current": count_processed,
            "total": total_imgs,
            "filename": img_name,
            "total_generated": total_generated
        }
        print(f"__AUG_PROGRESS__{json.dumps(progress_payload)}", flush=True)

    print(f"__AUG_FINISH__{json.dumps({'total_source': total_imgs, 'total_generated': total_generated, 'output_dir': output_dir})}", flush=True)

if __name__ == '__main__':
    main()
