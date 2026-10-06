# -*- coding: utf-8 -*-
"""
NovaVision - YOLO Training Bridge with real-time JSON metrics streaming and auto BOM sanitization
"""
import os
import sys
import gc
import faulthandler

# Immediately enable low-level fault handler to catch and diagnose any DLL/C++ access violations
try:
    faulthandler.enable()
except Exception:
    pass

import types
sys.modules['albumentations'] = None
sys.modules.setdefault('torchvision.models', types.ModuleType('torchvision.models'))

# Windows stability & PyTorch CUDA configuration
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

# Ensure PyTorch torch/lib and app DLLs are registered and PATH is sanitized from conflicting runtimes
from bridge_common import configure_windows_dll_path

configure_windows_dll_path()

import torch

# Headless matplotlib Agg backend configuration
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import logging
# Disable Ultralytics telemetry, PyPI pip update check, and auto-install to prevent cookiejar/requests crash
os.environ["YOLO_VERBOSE"] = "True"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "False"
try:
    import ultralytics.utils.checks as _checks
    _checks.check_pip_update_available = lambda: False
    _checks.check_latest_pypi_version = lambda *args, **kwargs: None
except Exception:
    pass


# Guard against torch._compile._disable_dynamo & torchvision triggering torch._dynamo & sympy C-stack crash
try:
    if hasattr(torch.optim.Optimizer.add_param_group, "__wrapped__"):
        torch.optim.Optimizer.add_param_group = torch.optim.Optimizer.add_param_group.__wrapped__
    if 'torch._dynamo' not in sys.modules:
        import types
        class _SafeDummyDynamo(types.ModuleType):
            __file__ = "torch/_dynamo/__init__.py"
            def __getattr__(self, name):
                if name == 'is_compiling':
                    return lambda: False
                if name == 'disable':
                    return lambda fn=None, recursive=True: fn if fn is not None else (lambda f: f)
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                return lambda *args, **kwargs: None

        _dummy_dynamo = _SafeDummyDynamo('torch._dynamo')
        _dummy_utils = _SafeDummyDynamo('torch._dynamo.utils')
        _dummy_utils.is_compile_supported = lambda device_type='': False
        _dummy_dynamo.utils = _dummy_utils
        sys.modules['torch._dynamo'] = _dummy_dynamo
        sys.modules['torch._dynamo.utils'] = _dummy_utils
    # Guard against torch.no_grad free variable '__class__' NameError on Windows Python 3.12
    torch.autograd.grad_mode.no_grad.__init__ = lambda self: setattr(self, 'prev', False)
    if hasattr(torch, 'no_grad'):
        torch.no_grad.__init__ = lambda self: setattr(self, 'prev', False)
except Exception:
    pass

# Guard against check_amp loading heavy duplicate models that cause 0xC0000005 crashes on Windows
def _safe_check_amp(m):
    try:
        from ultralytics.utils.torch_utils import autocast
        device = next(m.parameters()).device
        if device.type in {"cpu", "mps"}:
            return False
        with autocast(enabled=True, device=device.type):
            tx = torch.zeros((1, 3, 64, 64), device=device)
            tc = torch.nn.Conv2d(3, 3, 3).to(device)
            to = tc(tx)
            del tx, tc, to
        return True
    except Exception:
        return False

try:
    import ultralytics.utils.checks as checks_module
    checks_module.check_amp = _safe_check_amp
except Exception:
    pass

try:
    import ultralytics.engine.trainer as trainer_module
    trainer_module.check_amp = _safe_check_amp
except Exception:
    pass

# Monkey-patch IGNORE_CLS_DIRS to prevent output/runs/weights from being treated as classes
try:
    import ultralytics.data.utils as _du
    if hasattr(_du, 'IGNORE_CLS_DIRS'):
        _du.IGNORE_CLS_DIRS.update({"output", "outputs", "runs", "weights", ".cache", "__pycache__", "labels", "images"})
except Exception:
    pass
try:
    import ultralytics.data.split as _ds
    if hasattr(_ds, 'IGNORE_CLS_DIRS'):
        _ds.IGNORE_CLS_DIRS.update({"output", "outputs", "runs", "weights", "train", "val", "valid", "validation", "test", ".cache", "__pycache__", "labels", "images"})
except Exception:
    pass

# Guard against matplotlib / NumPy 2.x NaN crashes during metric curves plotting
def _safe_no_op(*args, **kwargs):
    return

try:
    import ultralytics.utils.metrics as _u_metrics
    _u_metrics.plot_mc_curve = _safe_no_op
    _u_metrics.plot_pr_curve = _safe_no_op
    if hasattr(_u_metrics, 'ConfusionMatrix'):
        _u_metrics.ConfusionMatrix.plot = lambda *a, **kw: None
except Exception:
    pass

# Guard against plotting functions triggering Windows STATUS_STACK_BUFFER_OVERRUN (0xC0000409) / ACCESS_VIOLATION
for _mod_name in ['ultralytics.utils.plotting', 'ultralytics.models.yolo.detect.train']:
    try:
        _m = __import__(_mod_name, fromlist=['plot_labels', 'plot_results', 'DetectionTrainer'])
        if hasattr(_m, 'plot_labels'):
            _m.plot_labels = _safe_no_op
        if hasattr(_m, 'plot_results'):
            _m.plot_results = _safe_no_op
        if hasattr(_m, 'DetectionTrainer'):
            _m.DetectionTrainer.plot_training_labels = _safe_no_op
    except Exception:
        pass

try:
    if 'trainer_module' in locals() and hasattr(trainer_module, 'BaseTrainer'):
        trainer_module.BaseTrainer.plot_metrics = _safe_no_op
except Exception:
    pass


# Guard against autobatch prober triggering Windows CUDA STATUS_ACCESS_VIOLATION (0xC0000005)
def _safe_check_train_batch_size(model, imgsz=640, amp=True, batch=-1, max_num_obj=1, dataset_size=0):
    return 8 if dataset_size <= 0 else min(8, dataset_size)

try:
    import ultralytics.utils.autobatch as autobatch_module
    autobatch_module.check_train_batch_size = _safe_check_train_batch_size
except Exception:
    pass

try:
    if 'trainer_module' in locals():
        trainer_module.check_train_batch_size = _safe_check_train_batch_size
except Exception:
    pass

# Guard against ModelEMA.update list accumulation/tensor attribute crash or foreach_lerp_ CUDA issues
def _safe_ema_update(self, model):
    if not getattr(self, "enabled", True):
        return
    self.updates += 1
    d = self.decay(self.updates)
    try:
        with torch.no_grad():
            from ultralytics.utils.torch_utils import unwrap_model
            msd = unwrap_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point and k in msd:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
    except Exception:
        pass

try:
    import ultralytics.utils.torch_utils as torch_utils_module
    torch_utils_module.ModelEMA.update = _safe_ema_update
    _orig_strip_optimizer = torch_utils_module.strip_optimizer
    def _safe_strip_optimizer(f, half=False, updates=None):
        try:
            import gc, time
            gc.collect()
            time.sleep(0.5)
            return _orig_strip_optimizer(f, half=half, updates=updates)
        except Exception as e:
            print(f"[Info] strip_optimizer note: {e}", flush=True)
            return {}
    torch_utils_module.strip_optimizer = _safe_strip_optimizer
except Exception:
    pass


import math

def _convert_label_line(line, task):
    """
    Intelligently converts a single label line to match the target task.
    """
    parts = line.strip().split()
    if not parts:
        return line
    n = len(parts)
    c_id = parts[0]
    try:
        coords = [float(p) for p in parts[1:]]
    except Exception:
        return line

    if task == 'obb':
        if n == 11:  # Pose format: cls cx cy w h k1x k1y v1 k2x k2y v2
            cx, cy, w, h = coords[0], coords[1], coords[2], coords[3]
            k1x, k1y, k2x, k2y = coords[4], coords[5], coords[7], coords[8]
            dx = k2x - k1x
            dy = k2y - k1y
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                ux = dx / dist
                uy = dy / dist
                vx = -uy
                vy = ux
                center_x = (k1x + k2x) / 2.0
                center_y = (k1y + k2y) / 2.0
                half_len = dist / 2.0
                half_w = min(w, h) / 2.0 if (w > 0 and h > 0) else 0.02
                p1x, p1y = center_x - half_len * ux - half_w * vx, center_y - half_len * uy - half_w * vy
                p2x, p2y = center_x + half_len * ux - half_w * vx, center_y + half_len * uy - half_w * vy
                p3x, p3y = center_x + half_len * ux + half_w * vx, center_y + half_len * uy + half_w * vy
                p4x, p4y = center_x - half_len * ux + half_w * vx, center_y - half_len * uy + half_w * vy
            else:
                half_w = (w if w > 0 else 0.04) / 2.0
                half_h = (h if h > 0 else 0.04) / 2.0
                p1x, p1y = cx - half_w, cy - half_h
                p2x, p2y = cx + half_w, cy - half_h
                p3x, p3y = cx + half_w, cy + half_h
                p4x, p4y = cx - half_w, cy + half_h
            pts = [max(0.0, min(1.0, p)) for p in (p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)]
            return f"{c_id} " + " ".join(f"{p:.6f}" for p in pts)
        elif n == 5:  # Standard Detect bbox: cls cx cy w h -> 4 corners
            cx, cy, w, h = coords[0], coords[1], coords[2], coords[3]
            half_w, half_h = w / 2.0, h / 2.0
            p1x, p1y = max(0.0, min(1.0, cx - half_w)), max(0.0, min(1.0, cy - half_h))
            p2x, p2y = max(0.0, min(1.0, cx + half_w)), max(0.0, min(1.0, cy - half_h))
            p3x, p3y = max(0.0, min(1.0, cx + half_w)), max(0.0, min(1.0, cy + half_h))
            p4x, p4y = max(0.0, min(1.0, cx - half_w)), max(0.0, min(1.0, cy + half_h))
            return f"{c_id} {p1x:.6f} {p1y:.6f} {p2x:.6f} {p2y:.6f} {p3x:.6f} {p3y:.6f} {p4x:.6f} {p4y:.6f}"
        elif n == 3:  # Point: cls cx cy
            cx, cy = coords[0], coords[1]
            hw, hh = 0.01, 0.01
            p1x, p1y = max(0.0, min(1.0, cx - hw)), max(0.0, min(1.0, cy - hh))
            p2x, p2y = max(0.0, min(1.0, cx + hw)), max(0.0, min(1.0, cy - hh))
            p3x, p3y = max(0.0, min(1.0, cx + hw)), max(0.0, min(1.0, cy + hh))
            p4x, p4y = max(0.0, min(1.0, cx - hw)), max(0.0, min(1.0, cy + hh))
            return f"{c_id} {p1x:.6f} {p1y:.6f} {p2x:.6f} {p2y:.6f} {p3x:.6f} {p3y:.6f} {p4x:.6f} {p4y:.6f}"
        elif n == 9:
            return line

    elif task == 'detect':
        if n == 11:  # Pose -> Detect: strip keypoints
            return " ".join(parts[:5])
        elif n == 9:  # OBB -> Detect: envelope bounding box
            xs = [coords[0], coords[2], coords[4], coords[6]]
            ys = [coords[1], coords[3], coords[5], coords[7]]
            min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
            min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            w = max_x - min_x
            h = max_y - min_y
            return f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        elif n == 3:  # Point -> Detect
            return f"{c_id} {coords[0]:.6f} {coords[1]:.6f} 0.020000 0.020000"
        elif n == 5:
            return line

    elif task == 'pose':
        if n == 11:
            return line
        elif n == 9:  # OBB -> Pose: convert 4 corners to bbox and 2 edge midpoints as keypoints
            xs = [coords[0], coords[2], coords[4], coords[6]]
            ys = [coords[1], coords[3], coords[5], coords[7]]
            min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
            min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            w = max_x - min_x
            h = max_y - min_y
            k1x = (coords[0] + coords[6]) / 2.0
            k1y = (coords[1] + coords[7]) / 2.0
            k2x = (coords[2] + coords[4]) / 2.0
            k2y = (coords[3] + coords[5]) / 2.0
            return f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {k1x:.6f} {k1y:.6f} 2 {k2x:.6f} {k2y:.6f} 2"
        elif n == 5:
            cx, cy, w, h = coords[0], coords[1], coords[2], coords[3]
            return f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2"
        elif n == 3:
            cx, cy = coords[0], coords[1]
            return f"{c_id} {cx:.6f} {cy:.6f} 0.020000 0.020000 {cx:.6f} {cy:.6f} 2"

    elif task == 'segment':
        if n == 5:  # Rect -> 4-point polygon
            cx, cy, w, h = coords[0], coords[1], coords[2], coords[3]
            half_w, half_h = w / 2.0, h / 2.0
            p1x, p1y = max(0.0, min(1.0, cx - half_w)), max(0.0, min(1.0, cy - half_h))
            p2x, p2y = max(0.0, min(1.0, cx + half_w)), max(0.0, min(1.0, cy - half_h))
            p3x, p3y = max(0.0, min(1.0, cx + half_w)), max(0.0, min(1.0, cy + half_h))
            p4x, p4y = max(0.0, min(1.0, cx - half_w)), max(0.0, min(1.0, cy + half_h))
            return f"{c_id} {p1x:.6f} {p1y:.6f} {p2x:.6f} {p2y:.6f} {p3x:.6f} {p3y:.6f} {p4x:.6f} {p4y:.6f}"
        elif n == 11:
            cx, cy, w, h = coords[0], coords[1], coords[2], coords[3]
            k1x, k1y, k2x, k2y = coords[4], coords[5], coords[7], coords[8]
            dx = k2x - k1x
            dy = k2y - k1y
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                ux = dx / dist
                uy = dy / dist
                vx = -uy
                vy = ux
                center_x = (k1x + k2x) / 2.0
                center_y = (k1y + k2y) / 2.0
                half_len = dist / 2.0
                half_w = min(w, h) / 2.0 if (w > 0 and h > 0) else 0.02
                p1x, p1y = center_x - half_len * ux - half_w * vx, center_y - half_len * uy - half_w * vy
                p2x, p2y = center_x + half_len * ux - half_w * vx, center_y + half_len * uy - half_w * vy
                p3x, p3y = center_x + half_len * ux + half_w * vx, center_y + half_len * uy + half_w * vy
                p4x, p4y = center_x - half_len * ux + half_w * vx, center_y - half_len * uy + half_w * vy
            else:
                half_w = (w if w > 0 else 0.04) / 2.0
                half_h = (h if h > 0 else 0.04) / 2.0
                p1x, p1y = cx - half_w, cy - half_h
                p2x, p2y = cx + half_w, cy - half_h
                p3x, p3y = cx + half_w, cy + half_h
                p4x, p4y = cx - half_w, cy + half_h
            pts = [max(0.0, min(1.0, p)) for p in (p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)]
            return f"{c_id} " + " ".join(f"{p:.6f}" for p in pts)

    return line

def sanitize_dataset(yaml_path, task='detect'):
    """
    Sanitizes dataset files:
    1. Removes UTF-8 BOM from all .txt labels and .yaml config files.
    2. Adapts cross-task label formats (e.g. Pose 11-values <-> OBB 9-values <-> Detect 5-values).
    3. Handles data.yaml kpt_shape configuration (uncomment for pose, comment out for others).
    4. Deletes all *.cache files to force Ultralytics to re-scan cleanly.
    """
    is_cls = task == 'classify' or (isinstance(yaml_path, str) and not yaml_path.endswith(('.yaml', '.yml')) and os.path.isdir(yaml_path))
    if is_cls:
        try:
            cls_dir = os.path.abspath(yaml_path) if os.path.isdir(yaml_path) else os.path.dirname(os.path.abspath(yaml_path))
            pseudo_prefixes = ('output', 'outputs', 'runs', 'weights', '.cache', '__pycache__')
            for check_root in (cls_dir, f"{cls_dir.rstrip(r'\/')}_split"):
                if not os.path.isdir(check_root):
                    continue
                for sub in ('train', 'val', 'valid', 'test'):
                    parent = os.path.join(check_root, sub)
                    if os.path.isdir(parent):
                        for item in list(os.listdir(parent)):
                            if item.lower().startswith(pseudo_prefixes):
                                target = os.path.join(parent, item)
                                if os.path.isdir(target):
                                    import shutil
                                    shutil.rmtree(target, ignore_errors=True)
                                    print(f"[分类数据集净化] 已自动清除子集中的伪分类目录: {target}", flush=True)
                for root, dirs, files in os.walk(check_root):
                    for f in files:
                        if f.endswith('.cache'):
                            try:
                                os.remove(os.path.join(root, f))
                            except Exception:
                                pass
        except Exception as e:
            print(f"[Warning] Classification dataset sanitization note: {e}", flush=True)
        return

    try:
        yaml_dir = os.path.abspath(yaml_path) if os.path.isdir(yaml_path) else os.path.dirname(os.path.abspath(yaml_path))
        converted_count = 0
        for root, dirs, files in os.walk(yaml_dir):
            for file in files:
                fpath = os.path.join(root, file)
                if file.endswith('.cache'):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                elif file.endswith('.txt') and file not in ('train.txt', 'val.txt', 'test.txt', 'classes.txt'):
                    try:
                        with open(fpath, 'r', encoding='utf-8-sig') as rf:
                            lines = rf.read().splitlines()
                        new_lines = []
                        file_modified = False
                        for line in lines:
                            conv = _convert_label_line(line, task)
                            if conv != line:
                                file_modified = True
                            new_lines.append(conv)
                        if file_modified:
                            converted_count += 1
                            with open(fpath, 'w', encoding='utf-8') as wf:
                                wf.write('\n'.join(new_lines) + ('\n' if new_lines else ''))
                    except Exception:
                        pass
                elif file.endswith('.txt') or file.endswith('.yaml') or file.endswith('.yml'):
                    try:
                        with open(fpath, 'r', encoding='utf-8-sig') as rf:
                            text = rf.read()
                        with open(fpath, 'w', encoding='utf-8') as wf:
                            wf.write(text)
                    except Exception:
                        pass
        if converted_count > 0:
            print(f"[自适应适配] 已自动将 {converted_count} 个标注文件适配至目标任务规格 ({task.upper()})！", flush=True)

        # Synchronize data.yaml kpt_shape according to task
        if os.path.isfile(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                y_lines = f.readlines()
            new_y_lines = []
            yaml_changed = False
            has_kpt_shape = False
            for yl in y_lines:
                s = yl.strip()
                if s.startswith('kpt_shape:') or s.startswith('# kpt_shape:'):
                    has_kpt_shape = True
                    if task != 'pose':
                        if not s.startswith('#'):
                            new_y_lines.append(f"# {yl.lstrip()}")
                            yaml_changed = True
                        else:
                            new_y_lines.append(yl)
                    else:
                        if s.startswith('#'):
                            new_y_lines.append(s.lstrip('# ').strip() + '\n')
                            yaml_changed = True
                        else:
                            new_y_lines.append(yl)
                else:
                    new_y_lines.append(yl)
            if task == 'pose' and not has_kpt_shape:
                new_y_lines.append("kpt_shape: [2, 3]\n")
                yaml_changed = True
            if yaml_changed:
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_y_lines)

        # Classification dataset sanitization: purge pseudo-class directories (e.g. output, runs, weights)
        if task == 'classify' or os.path.isdir(yaml_path):
            cls_dir = os.path.abspath(yaml_path) if os.path.isdir(yaml_path) else os.path.dirname(os.path.abspath(yaml_path))
            pseudo_names = {'output', 'outputs', 'runs', 'weights', '.cache', '__pycache__'}
            for sub in ('train', 'val', 'valid', 'test'):
                parent = os.path.join(cls_dir, sub)
                if os.path.isdir(parent):
                    for pseudo in pseudo_names:
                        target = os.path.join(parent, pseudo)
                        if os.path.isdir(target):
                            import shutil
                            shutil.rmtree(target, ignore_errors=True)
                            print(f"[分类数据集净化] 已自动清除子集中的伪分类目录: {target}", flush=True)
            # Also clean up in {cls_dir}_split
            split_dir = f"{cls_dir}_split"
            if os.path.isdir(split_dir):
                for sub in ('train', 'val', 'valid', 'test'):
                    parent = os.path.join(split_dir, sub)
                    if os.path.isdir(parent):
                        for pseudo in pseudo_names:
                            target = os.path.join(parent, pseudo)
                            if os.path.isdir(target):
                                import shutil
                                shutil.rmtree(target, ignore_errors=True)
    except Exception as e:
        print(f"[Warning] Dataset sanitization note: {e}", flush=True)
    
    # Auto-repair train.txt or val.txt if missing/empty
    try:
        if os.path.isfile(yaml_path):
            yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
            with open(yaml_path, 'r', encoding='utf-8', errors='ignore') as f:
                ycontent = f.read()
            if 'train.txt' in ycontent:
                train_txt = os.path.join(yaml_dir, 'train.txt')
                val_txt = os.path.join(yaml_dir, 'val.txt')
                if not os.path.isfile(train_txt) or os.path.getsize(train_txt) == 0:
                    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                    all_imgs = []
                    for root, dirs, files in os.walk(yaml_dir):
                        if any(x in root.lower() for x in ['output', 'runs', '.git', '__pycache__', 'weights']):
                            continue
                        for file in files:
                            if file.lower().endswith(valid_exts):
                                rel = os.path.relpath(os.path.join(root, file), yaml_dir).replace('\\', '/')
                                all_imgs.append(f"./{rel}")
                    if all_imgs:
                        print(f"[自检修复] 检测到 train.txt 索引清单缺失，已自动扫描并索引 {len(all_imgs)} 张图片...", flush=True)
                        with open(train_txt, 'w', encoding='utf-8') as tf:
                            tf.write('\n'.join(all_imgs) + '\n')
                        if not os.path.isfile(val_txt) or os.path.getsize(val_txt) == 0:
                            v_count = max(1, len(all_imgs) // 5)
                            with open(val_txt, 'w', encoding='utf-8') as vf:
                                vf.write('\n'.join(all_imgs[:v_count]) + '\n')
    except Exception as e:
        print(f"[自检提醒] 数据集清单校验提示: {e}", flush=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v_lower = str(v).lower()
    if v_lower in ('yes', 'true', 't', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', '0'):
        return False
    else:
        return True

def is_yolov5_model(model_path):
    if not model_path:
        return False
    fn = os.path.basename(str(model_path)).lower()
    if fn.startswith('yolov5') or 'yolov5' in fn:
        return True
    if str(model_path).endswith('.pt') and os.path.isfile(model_path):
        try:
            import zipfile
            with zipfile.ZipFile(model_path, 'r') as z:
                names = z.namelist()
                pkl_name = next((n for n in names if n.endswith('data.pkl')), None)
                if pkl_name:
                    raw = z.read(pkl_name)
                    if b'models.yolo' in raw or b'models.common' in raw:
                        return True
        except Exception:
            pass
    return False

def is_vajra_model(model_path):
    if not model_path:
        return False
    fn = os.path.basename(str(model_path)).lower()
    if 'vajra' in fn:
        return True
    if str(model_path).endswith('.pt') and os.path.isfile(model_path):
        try:
            import zipfile
            with zipfile.ZipFile(model_path, 'r') as z:
                names = z.namelist()
                pkl_name = next((n for n in names if n.endswith('data.pkl')), None)
                if pkl_name:
                    raw = z.read(pkl_name)
                    if any(k in raw for k in [b'vajra.nn', b'vajra.models', b'vajra.core', b'Merudanda', b'VajraV1', b'VajraV2']):
                        return True
        except Exception:
            pass
    return False

def sync_vajra_checkpoints(save_dir):
    if not save_dir:
        return
    weights_dir = os.path.join(str(save_dir), 'weights')
    if not os.path.isdir(weights_dir):
        return
    import shutil
    try:
        best_candidates = [f for f in os.listdir(weights_dir) if f.startswith('best') and f.endswith('.pt') and f != 'best.pt']
        if best_candidates:
            best_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(weights_dir, x)), reverse=True)
            target_best = os.path.join(weights_dir, 'best.pt')
            shutil.copy2(os.path.join(weights_dir, best_candidates[0]), target_best)
    except Exception:
        pass

    try:
        last_candidates = [f for f in os.listdir(weights_dir) if f.startswith('last') and f.endswith('.pt') and f != 'last.pt']
        if last_candidates:
            last_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(weights_dir, x)), reverse=True)
            target_last = os.path.join(weights_dir, 'last.pt')
            shutil.copy2(os.path.join(weights_dir, last_candidates[0]), target_last)
    except Exception:
        pass

def train_yolov5(args, model_to_load):
    import yaml
    from pathlib import Path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(script_dir), "yolov5"),
        os.path.join(script_dir, "yolov5"),
        r"F:\Yolo26\Debug\yolov5",
        r"F:\Yolo26\Debug\App_NovaVision\yolov5",
        r"F:\YOLO_ training_tool\yolo\model\yolov5"
    ]
    yolov5_dir = None
    for cand in candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "train.py")):
            yolov5_dir = cand
            break

    if not yolov5_dir:
        print(f"__TRAIN_ERROR__未找到 YOLOv5 核心源码引擎目录！", flush=True)
        sys.exit(1)

    if yolov5_dir not in sys.path:
        sys.path.insert(0, yolov5_dir)

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["COMET_MODE"] = "DISABLED"

    try:
        import train as yolov5_train
        from utils.callbacks import Callbacks
        from utils.general import increment_path
        from utils.torch_utils import select_device
    except Exception as e:
        print(f"__TRAIN_ERROR__导入 YOLOv5 训练模块失败: {e}", flush=True)
        sys.exit(1)

    # Sanitize dataset
    sanitize_dataset(args.data, task='detect')

    # Prepare directories and workers
    safe_workers = 0 if sys.platform.startswith('win') else args.workers
    exp_dir = os.path.join(args.project, args.name)
    os.makedirs(exp_dir, exist_ok=True)

    # Build YOLOv5 opt
    opt = yolov5_train.parse_opt(known=True)
    opt.weights = str(model_to_load)
    opt.cfg = ''
    opt.data = str(args.data)
    opt.epochs = args.epochs
    opt.batch_size = args.batch
    opt.imgsz = args.imgsz
    opt.device = str(args.device) if (args.device and args.device != 'auto') else ''
    opt.workers = safe_workers
    opt.project = str(args.project)
    opt.name = str(args.name)
    opt.exist_ok = args.exist_ok
    opt.patience = args.patience
    opt.save_period = args.save_period if args.save_period > 0 else -1
    opt.noplots = not getattr(args, 'plots', True)
    opt.noval = False
    opt.nosave = False
    opt.rect = False
    opt.cache = 'ram' if str(args.cache).lower() in ('ram',) else (args.cache if str(args.cache).lower() == 'disk' else None)

    opt_str = (args.optimizer or 'auto').lower()
    if opt_str == 'adam':
        opt.optimizer = 'Adam'
    elif opt_str == 'adamw':
        opt.optimizer = 'AdamW'
    else:
        opt.optimizer = 'SGD'

    # Hyperparameters
    hyp_candidate = os.path.join(yolov5_dir, "data", "hyps", "hyp.scratch-low.yaml")
    hyp = {}
    if os.path.isfile(hyp_candidate):
        try:
            with open(hyp_candidate, "r", encoding="utf-8", errors="ignore") as f:
                hyp = yaml.safe_load(f) or {}
        except Exception:
            pass

    if args.lr0 > 0:
        hyp['lr0'] = args.lr0
    if hasattr(args, 'mosaic') and args.mosaic is not None:
        hyp['mosaic'] = args.mosaic
    if hasattr(args, 'mixup') and args.mixup is not None:
        hyp['mixup'] = args.mixup
    if hasattr(args, 'fliplr') and args.fliplr is not None:
        hyp['fliplr'] = args.fliplr
    if hasattr(args, 'flipud') and args.flipud is not None:
        hyp['flipud'] = args.flipud
    if hasattr(args, 'degrees') and args.degrees is not None:
        hyp['degrees'] = args.degrees
    if hasattr(args, 'scale') and args.scale is not None:
        hyp['scale'] = args.scale
    if hasattr(args, 'translate') and args.translate is not None:
        hyp['translate'] = args.translate
    if hasattr(args, 'hsv_h') and args.hsv_h is not None:
        hyp['hsv_h'] = args.hsv_h
    if hasattr(args, 'hsv_s') and args.hsv_s is not None:
        hyp['hsv_s'] = args.hsv_s
    if hasattr(args, 'hsv_v') and args.hsv_v is not None:
        hyp['hsv_v'] = args.hsv_v

    hyp_file = os.path.join(exp_dir, "hyp.scratch.yaml")
    with open(hyp_file, "w", encoding="utf-8") as f:
        yaml.dump(hyp, f)
    opt.hyp = hyp_file

    if args.resume or (model_to_load and os.path.basename(model_to_load).lower() == 'last.pt'):
        opt.resume = str(model_to_load) if os.path.isfile(model_to_load) else True
        opt.save_dir = str(Path(model_to_load).parent.parent)
        print(f"[断点续训] YOLOv5 已开启 Resume 模式，将从检查点 '{model_to_load}' 恢复优化器与训练状态！", flush=True)
    else:
        opt.resume = False
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Register callbacks
    callbacks = Callbacks()
    current_epoch_ref = [0]

    def yolov5_on_pretrain_routine_end(labels=None, names=None):
        print(f"[就绪] YOLOv5 模型与优化器初始化完成，开始执行第 1 轮模型训练...", flush=True)

    def yolov5_on_train_epoch_start():
        current_epoch_ref[0] += 1
        print(f"[Epoch {current_epoch_ref[0]}/{args.epochs}] 正在进行 GPU 批次前向计算与梯度反向传播...", flush=True)

    def yolov5_on_val_start():
        print(f"[Epoch {current_epoch_ref[0]}/{args.epochs}] 批次训练完成，正在计算验证集 mAP 指标...", flush=True)

    def yolov5_on_fit_epoch_end(log_vals, epoch, best_fitness, fi):
        try:
            box_loss = round(float(log_vals[0]), 4) if len(log_vals) > 0 else 0.0
            obj_loss = round(float(log_vals[1]), 4) if len(log_vals) > 1 else 0.0
            cls_loss = round(float(log_vals[2]), 4) if len(log_vals) > 2 else 0.0
            map50 = round(float(log_vals[5]), 4) if len(log_vals) > 5 else 0.0
            map50_95 = round(float(log_vals[6]), 4) if len(log_vals) > 6 else 0.0
            lr = round(float(log_vals[7]), 6) if len(log_vals) > 7 else 0.0

            payload = {
                "epoch": epoch + 1,
                "total_epochs": args.epochs,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": obj_loss,
                "map50": map50,
                "map50_95": map50_95,
                "lr": lr,
                "is_classification": False
            }
            print(f"__EPOCH_METRIC__{json.dumps(payload)}", flush=True)
        except Exception:
            pass

    callbacks.register_action("on_pretrain_routine_end", callback=yolov5_on_pretrain_routine_end)
    callbacks.register_action("on_train_epoch_start", callback=yolov5_on_train_epoch_start)
    callbacks.register_action("on_val_start", callback=yolov5_on_val_start)
    callbacks.register_action("on_fit_epoch_end", callback=yolov5_on_fit_epoch_end)

    print("__TRAIN_START__", flush=True)

    try:
        device = select_device(opt.device, batch_size=opt.batch_size)
        results = yolov5_train.train(hyp, opt, device, callbacks)
        save_dir = opt.save_dir
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        # Fallback guarantee: ensure results.png is generated if results.csv exists
        try:
            results_csv = os.path.join(str(save_dir), "results.csv")
            results_png = os.path.join(str(save_dir), "results.png")
            if os.path.isfile(results_csv) and not os.path.isfile(results_png):
                from utils.plots import plot_results
                plot_results(file=results_csv, dir=str(save_dir))
        except Exception:
            pass
        print(f"__TRAIN_FINISHED__{json.dumps({'save_dir': str(save_dir), 'best_pt': best_pt})}", flush=True)
    except Exception as ex:
        print(f"__TRAIN_ERROR__{str(ex)}", flush=True)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NovaVision YOLO Training Bridge")
    parser.add_argument('--model', type=str, required=True, help='Model file path (e.g. yolo26n.pt)')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (SGD, Adam, AdamW, auto)')
    parser.add_argument('--device', type=str, default=None, help='Device (0, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project dir')
    parser.add_argument('--name', type=str, default='train', help='Run name')
    parser.add_argument('--exist_ok', type=str2bool, default=True, help='Overwrite existing')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Disable mosaic for last N epochs')
    
    # Augmentations
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=0.0, help='MixUp augmentation')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip probability')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation degrees')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale augmentation')
    parser.add_argument('--translate', type=float, default=0.1, help='Translate augmentation')
    parser.add_argument('--erasing', type=float, default=0.0, help='Random erasing probability')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-Hue')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-Saturation')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-Value')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume training from last checkpoint (last.pt)')
    parser.add_argument('--cache', type=str, default='false', help='Cache dataset images (ram, disk, false)')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every X epochs (<=0 disables periodic save)')
    parser.add_argument('--amp', type=str2bool, default=True, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--plots', type=str2bool, default=True, help='Generate training and validation plots and curve charts')
    
    args, _ = parser.parse_known_args()
    
    # Auto Batch: resolve batch <= 0 to safe batch size without Windows CUDA memory probe crash
    if args.batch <= 0:
        print(f"[自动批次 Auto] 检测到批次参数为 0 (自动模式)，系统已自动选用最佳安全批次: 8", flush=True)
        args.batch = 8
    
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
        except Exception:
            pass

    model_to_load = args.model
    if not os.path.isfile(model_to_load):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cand_dirs = [
            os.path.join(os.path.dirname(script_dir), "models"),
            os.path.join(os.path.dirname(script_dir), "App_NovaVision", "models"),
            os.path.join(script_dir, "models"),
            r"F:\Yolo26\Debug\App_NovaVision\models",
            r"F:\Yolo26\Debug\models"
        ]
        base_name = os.path.basename(model_to_load)
        for cd in cand_dirs:
            p = os.path.join(cd, base_name)
            if os.path.isfile(p):
                model_to_load = p
                print(f"[模型定位] 已自动匹配本地模型权重: {model_to_load}", flush=True)
                break

    if args.resume and (not model_to_load or os.path.basename(model_to_load).lower() != 'last.pt'):
        candidate1 = os.path.join(args.project, args.name, 'weights', 'last.pt')
        candidate2 = os.path.join(args.project, 'weights', 'last.pt')
        if os.path.isfile(candidate1):
            model_to_load = candidate1
            print(f"[断点续训] 自动定位断点检查点: {model_to_load}", flush=True)
        elif os.path.isfile(candidate2):
            model_to_load = candidate2
            print(f"[断点续训] 自动定位断点检查点: {model_to_load}", flush=True)

    # 预训练权重不能作为断点；在加载模型或改写数据集之前拒绝错误配置。
    should_resume = args.resume or os.path.basename(model_to_load).lower() == 'last.pt'
    if should_resume:
        if os.path.basename(model_to_load).lower() != 'last.pt' or not os.path.isfile(model_to_load):
            print(
                "__TRAIN_ERROR__断点续训需要有效的 weights/last.pt；当前选择的是预训练权重。"
                "若要开始新训练，请取消“断点继续训练”。",
                flush=True,
            )
            sys.exit(2)
        try:
            checkpoint = torch.load(model_to_load, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict) or checkpoint.get('epoch', -1) < 0 or checkpoint.get('optimizer') is None:
                raise ValueError('缺少有效轮次或优化器状态')
            del checkpoint
        except Exception as exc:
            print(f"__TRAIN_ERROR__断点文件不能用于续训：{exc}", flush=True)
            sys.exit(2)

    if is_yolov5_model(model_to_load):
        print(f"[架构识别] 检测到 YOLOv5 架构模型: {model_to_load}，已自动启用 YOLOv5 原生训练引擎...", flush=True)
        train_yolov5(args, model_to_load)
        return

    is_vajra = is_vajra_model(model_to_load)
    if is_vajra:
        try:
            import vajra.checks as _v_chk
            _v_chk.check_amp = _safe_check_amp
        except Exception:
            pass
        try:
            import vajra.core.trainer as _v_tr
            _v_tr.check_amp = _safe_check_amp
            _v_tr.strip_optimizer = lambda *a, **k: None
        except Exception:
            pass
        from vajra import Vajra
        print(f"[架构识别] 检测到 VajraV1 (VayuAI) 架构模型: {model_to_load}，已自动启用 Vajra 专有训练引擎...", flush=True)
        model = Vajra(model_to_load)
    else:
        from ultralytics import YOLO
        print(f"Initializing YOLO model from {model_to_load}...", flush=True)
        model = YOLO(model_to_load)
    task_name = getattr(model, 'task', 'detect')
    is_cls = task_name == 'classify'

    # If classification task:
    if is_cls:
        if os.path.isfile(args.data):
            args.data = os.path.dirname(os.path.abspath(args.data))
            print(f"[分类数据集自适应] 识别到分类任务传入了配置文件路径，已自动切换为数据集目录: '{args.data}'", flush=True)

        if os.path.isdir(args.data):
            from pathlib import Path
            d_path = Path(os.path.abspath(args.data))
            d_name = d_path.name.lower()
            IGNORE_CLS_DIRS = {"output", "outputs", "runs", "weights", "train", "val", "valid", "validation", "test", ".cache", "__pycache__", "labels", "images"}

            # 1. 误选了子集目录 (如 dataset/train 或 dataset/val)
            if d_name in ('train', 'val', 'valid'):
                parent_d = d_path.parent
                if (parent_d / 'train').is_dir() or (parent_d / 'val').is_dir():
                    print(f"[分类数据集自适应] 检测到选择的是子集目录 '{d_path}'，已自动纠正为根数据集目录 '{parent_d}'", flush=True)
                    args.data = str(parent_d)
                    d_path = parent_d

            # 2. 检查当前目录是否已包含预拆分的 train 或 val 目录
            has_train = (d_path / 'train').is_dir()
            has_val = (d_path / 'val').is_dir() or (d_path / 'validation').is_dir() or (d_path / 'valid').is_dir()

            if has_train or has_val:
                # 已经是完整的拆分数据集目录，直接使用，严禁回溯至上级目录
                args.data = str(d_path)
            else:
                # 检查是否存在已预拆分的同名 _split 目录
                split_cand = Path(f"{str(d_path).rstrip(r'\/')}_split")
                if (split_cand / 'train').is_dir() and (split_cand / 'val').is_dir():
                    print(f"[分类数据集自适应] 检测到已存在拆分数据集目录 '{split_cand}'，已自动接入该拆分目录！", flush=True)
                    args.data = str(split_cand)
                    d_path = split_cand
                else:
                    # 3. 误选了单个类别文件夹 (如 fruits_cls/apple)，只有在非系统/非桌面根目录时才回溯到父目录
                    sub_dirs = [d for d in d_path.iterdir() if d.is_dir() and d.name.lower() not in IGNORE_CLS_DIRS and not d.name.startswith('.')]
                    if len(sub_dirs) == 0:
                        parent_d = d_path.parent
                        system_dirs = {Path.home(), Path.home() / 'Desktop', Path.home() / 'Downloads', Path.home() / 'Documents', Path(parent_d.anchor)}
                        if parent_d not in system_dirs:
                            parent_sub_dirs = [d for d in parent_d.iterdir() if d.is_dir() and d.name.lower() not in IGNORE_CLS_DIRS and not d.name.startswith('.')]
                            if len(parent_sub_dirs) >= 2:
                                has_imgs = any(len([f for f in d.glob('*.*') if f.suffix[1:].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'webp'}]) > 0 for d in parent_sub_dirs)
                                if has_imgs:
                                    names_str = ', '.join([d.name for d in parent_sub_dirs[:6]]) + ('...' if len(parent_sub_dirs) > 6 else '')
                                    print(f"[分类数据集智能识别] 检测到您当前选择的是单个类别子目录 '{d_path.name}'，系统已自动重定向至完整分类数据集根目录: '{parent_d}' (共 {len(parent_sub_dirs)} 个类别: {names_str})", flush=True)
                                    args.data = str(parent_d)
                                    if os.path.abspath(args.project) == str(d_path):
                                        args.project = str(parent_d)

    # Auto sanitize dataset BOM & adapt point labels & clear cached files before training
    sanitize_dataset(args.data, task=task_name)
    
    def on_pretrain_routine_end(trainer):
        print(f"[就绪] 模型与优化器初始化完成，开始执行第 1 轮模型训练...", flush=True)

    current_training_epoch = 1
    total_training_epochs = args.epochs
    last_reported_epoch = -1

    def on_train_epoch_start(trainer):
        nonlocal current_training_epoch, total_training_epochs
        try:
            current_training_epoch = getattr(trainer, 'epoch', 0) + 1
            total_training_epochs = getattr(trainer, 'epochs', args.epochs)
            print(f"[Epoch {current_training_epoch}/{total_training_epochs}] 正在进行 GPU 批次前向计算与梯度反向传播...", flush=True)
        except Exception:
            pass

    def on_val_start(validator):
        try:
            print(f"[Epoch {current_training_epoch}/{total_training_epochs}] 批次训练完成，正在计算验证集 mAP 指标...", flush=True)
        except Exception:
            pass

    # Define custom callback to stream epoch metrics safely to C#
    def on_fit_epoch_end(trainer):
        nonlocal last_reported_epoch
        try:
            cur_epoch = getattr(trainer, 'epoch', 0) + 1
            tot_epochs = getattr(trainer, 'epochs', args.epochs)

            # Skip final validation of best.pt (which runs after training completes at cur_epoch > tot_epochs)
            # or duplicate callbacks for the same epoch
            if cur_epoch > tot_epochs or cur_epoch == last_reported_epoch:
                return
            last_reported_epoch = cur_epoch

            task_name = getattr(trainer, 'args', None)
            trainer_is_cls = is_cls or (task_name and getattr(task_name, 'task', '') == 'classify')

            # 1. Loss extraction: Use Ultralytics standard label_loss_items which returns native Python floats
            loss_dict = {}
            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                try:
                    labeled = trainer.label_loss_items(trainer.tloss, prefix="train")
                    if isinstance(labeled, dict):
                        for k, v in labeled.items():
                            kl = str(k).lower()
                            val = round(float(v), 4)
                            if 'box' in kl:
                                loss_dict['box_loss'] = val
                            elif 'cls' in kl:
                                loss_dict['cls_loss'] = val
                            elif 'dfl' in kl:
                                loss_dict['dfl_loss'] = val
                            elif 'seg' in kl:
                                loss_dict['seg_loss'] = val
                            elif 'loss' in kl:
                                loss_dict['cls_loss'] = val
                except Exception:
                    pass

            # Robust fallback for classification loss
            if trainer_is_cls and 'cls_loss' not in loss_dict:
                if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                    try:
                        if isinstance(trainer.tloss, dict):
                            val = next(iter(trainer.tloss.values()))
                            loss_dict['cls_loss'] = round(float(val), 4)
                        elif hasattr(trainer.tloss, 'item'):
                            loss_dict['cls_loss'] = round(float(trainer.tloss.item()), 4)
                        elif isinstance(trainer.tloss, (float, int)):
                            loss_dict['cls_loss'] = round(float(trainer.tloss), 4)
                    except Exception:
                        pass
                if 'cls_loss' not in loss_dict and hasattr(trainer, 'loss') and trainer.loss is not None:
                    try:
                        if hasattr(trainer.loss, 'item'):
                            loss_dict['cls_loss'] = round(float(trainer.loss.item()), 4)
                        elif isinstance(trainer.loss, (float, int)):
                            loss_dict['cls_loss'] = round(float(trainer.loss), 4)
                    except Exception:
                        pass

            # If segmentation task, map seg_loss to dfl_loss so the UI displays mask loss curve
            if 'dfl_loss' not in loss_dict and 'seg_loss' in loss_dict:
                loss_dict['dfl_loss'] = loss_dict['seg_loss']

            # 2. Metric extraction: from trainer.metrics (native Python floats)
            map50 = 0.0
            map50_95 = 0.0
            if hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
                m = trainer.metrics
                if trainer_is_cls:
                    try:
                        map50 = round(float(m.get('metrics/accuracy_top1', m.get('accuracy_top1', 0.0))), 4)
                        map50_95 = round(float(m.get('metrics/accuracy_top5', m.get('accuracy_top5', 0.0))), 4)
                    except Exception:
                        pass
                else:
                    # For segmentation, mask mAP (M) is the primary evaluation metric; otherwise box mAP (B) or pose (P)
                    for key50, key95 in [
                        ('metrics/mAP50(Box)', 'metrics/mAP50-95(Box)'),
                        ('metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)'),
                        ('metrics/mAP50(Pose)', 'metrics/mAP50-95(Pose)'),
                        ('metrics/mAP50(M)', 'metrics/mAP50-95(M)'),
                        ('metrics/mAP50(B)', 'metrics/mAP50-95(B)'),
                        ('metrics/mAP50(P)', 'metrics/mAP50-95(P)'),
                        ('metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'),
                        ('metrics/mAP50', 'metrics/mAP50-95')
                    ]:
                        if key50 in m:
                            try:
                                map50 = round(float(m[key50]), 4)
                                map50_95 = round(float(m.get(key95, 0.0)), 4)
                                break
                            except Exception:
                                pass

                    if map50 == 0.0 and isinstance(m, dict):
                        for k, v in m.items():
                            kl = str(k).lower()
                            if 'map50-95' in kl or 'map_0.5:0.95' in kl:
                                try: map50_95 = round(float(v), 4)
                                except Exception: pass
                            elif 'map50' in kl or 'map_0.5' in kl:
                                try: map50 = round(float(v), 4)
                                except Exception: pass

            # 3. Learning rate: from trainer.lr or trainer.optimizer
            lr = 0.0
            if hasattr(trainer, 'lr') and isinstance(trainer.lr, dict) and trainer.lr:
                try:
                    lr = round(float(next(iter(trainer.lr.values()))), 6)
                except Exception:
                    pass
            elif hasattr(trainer, 'optimizer') and trainer.optimizer:
                try:
                    lr = round(float(trainer.optimizer.param_groups[0]['lr']), 6)
                except Exception:
                    pass

            if is_vajra and hasattr(trainer, 'save_dir'):
                sync_vajra_checkpoints(trainer.save_dir)

            payload = {
                "epoch": cur_epoch,
                "total_epochs": tot_epochs,
                "box_loss": loss_dict.get('box_loss', 0.0),
                "cls_loss": loss_dict.get('cls_loss', 0.0),
                "dfl_loss": loss_dict.get('dfl_loss', 0.0),
                "map50": map50,
                "map50_95": map50_95,
                "lr": lr,
                "is_classification": trainer_is_cls
            }
            print(f"__EPOCH_METRIC__{json.dumps(payload)}", flush=True)
        except Exception:
            pass

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_val_start", on_val_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # On Windows, PyTorch DataLoader multiprocessing with workers > 0 can deadlock when closing mosaic at epoch (epochs - close_mosaic)
    safe_workers = 0 if sys.platform.startswith('win') else args.workers

    cache_arg = str(args.cache).strip().lower()
    if cache_arg in ('ram', 'true', '1'):
        cache_mode = 'ram'
    elif cache_arg in ('disk',):
        cache_mode = 'disk'
    else:
        cache_mode = False

    opt_val = args.optimizer or 'auto'
    if opt_val.lower() == 'musgd':
        model_fn = os.path.basename(model_to_load).lower()
        if not ('26' in model_fn or 'yolo26' in model_fn or 'yoloe-26' in model_fn):
            print(f"[优化器适配] 检测到当前模型 ({model_fn}) 非 YOLO26 架构，MuSGD 专属 YOLO26，已自动适配切换为: auto", flush=True)
            opt_val = 'auto'

    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'optimizer': opt_val,
        'device': args.device if (args.device and args.device != 'auto') else None,
        'workers': safe_workers,
        'cache': cache_mode,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'close_mosaic': args.close_mosaic,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'fliplr': args.fliplr,
        'flipud': args.flipud,
        'degrees': args.degrees,
        'scale': args.scale,
        'translate': args.translate,
        'erasing': args.erasing,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'patience': args.patience,
        'save_period': args.save_period if args.save_period > 0 else -1,
        'amp': args.amp,
        'verbose': True,
        'plots': getattr(args, 'plots', False)
    }
    if is_vajra:
        train_kwargs['img_size'] = args.imgsz
    
    if should_resume:
        train_kwargs['resume'] = str(model_to_load) if os.path.isfile(model_to_load) else True
        print(f"[断点续训] 已开启 Resume 模式，将从模型检查点 '{model_to_load}' 恢复优化器与训练状态！", flush=True)
    
    print("__TRAIN_START__", flush=True)
    try:
        results = model.train(**train_kwargs)
        save_dir = getattr(trainer_obj := getattr(model, 'trainer', None), 'save_dir', args.project)
        if is_vajra:
            sync_vajra_checkpoints(save_dir)
        best_pt = os.path.join(str(save_dir), 'weights', 'best.pt') if save_dir else ""
        if not os.path.isfile(best_pt) and save_dir:
            w_dir = os.path.join(str(save_dir), 'weights')
            if os.path.isdir(w_dir):
                b_cands = [f for f in os.listdir(w_dir) if f.startswith('best') and f.endswith('.pt')]
                if b_cands:
                    b_cands.sort(key=lambda x: os.path.getmtime(os.path.join(w_dir, x)), reverse=True)
                    best_pt = os.path.join(w_dir, b_cands[0])

        print(f"__TRAIN_FINISHED__{json.dumps({'save_dir': str(save_dir), 'best_pt': best_pt})}", flush=True)
        sys.stderr.flush()
        return
    except Exception as ex:
        import traceback
        traceback.print_exc()
        tb = traceback.format_exc()
        print(f"__TRAIN_ERROR__{str(ex)}\n{tb}", flush=True)
        sys.stderr.flush()
        sys.exit(1)

if __name__ == '__main__':
    main()
