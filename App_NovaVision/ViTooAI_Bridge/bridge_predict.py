# -*- coding: utf-8 -*-
"""
NovaVision - YOLO Python Inference & Auto-Annotation Bridge
Supports Detect, Segment, Classify, OBB with JSON output.
"""
import sys
import os
import faulthandler

try:
    faulthandler.enable()
except Exception:
    pass

import types
sys.modules['albumentations'] = None
sys.modules.setdefault('torchvision.models', types.ModuleType('torchvision.models'))

from bridge_common import configure_windows_dll_path

configure_windows_dll_path()

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

import json
import argparse
import time
from pathlib import Path
import torch
from ultralytics import YOLO

class SimpleYOLOv5Result:
    def __init__(self, path, orig_shape, boxes_xyxy, boxes_conf, boxes_cls, names, speed):
        self.path = path
        self.orig_shape = orig_shape
        self.speed = speed
        self.names = names
        self.obb = None
        self.masks = None
        self.probs = None
        self.keypoints = None
        
        class Boxes:
            def __init__(self):
                self.xyxy = None
                self.conf = None
                self.cls = None
                self.xywhn = None
            def __len__(self):
                return len(self.xyxy) if self.xyxy is not None else 0
        b = Boxes()
        b.xyxy = torch.tensor(boxes_xyxy) if len(boxes_xyxy) else torch.empty((0, 4))
        b.conf = torch.tensor(boxes_conf) if len(boxes_conf) else torch.empty((0,))
        b.cls = torch.tensor(boxes_cls) if len(boxes_cls) else torch.empty((0,))
        
        h, w = orig_shape
        if len(boxes_xyxy):
            norm_boxes = []
            for (x1, y1, x2, y2) in boxes_xyxy:
                cx = (x1 + x2) / 2.0 / max(1, w)
                cy = (y1 + y2) / 2.0 / max(1, h)
                bw = (x2 - x1) / max(1, w)
                bh = (y2 - y1) / max(1, h)
                norm_boxes.append([cx, cy, bw, bh])
            b.xywhn = torch.tensor(norm_boxes)
        else:
            b.xywhn = torch.empty((0, 4))
        self.boxes = b

def ensure_yolov5_on_path():
    bridge_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(bridge_dir)
    possible_roots = [
        os.path.join(root_dir, "yolov5"),
        os.path.join(bridge_dir, "yolov5"),
        r"F:\Yolo26\Debug\yolov5",
        r"F:\Yolo26\Debug\App_NovaVision\yolov5",
        r"F:\YOLO_ training_tool\yolo\model\yolov5"
    ]
    for pr in possible_roots:
        if os.path.isdir(pr) and os.path.isdir(os.path.join(pr, "models")):
            if pr in sys.path:
                sys.path.remove(pr)
            sys.path.insert(0, pr)
            # If sys.modules['models'] was already bound to a directory without common.py, purge it!
            if 'models' in sys.modules:
                mod = sys.modules['models']
                if not hasattr(mod, '__file__') or not mod.__file__ or 'yolov5' not in str(mod.__file__):
                    for k in list(sys.modules.keys()):
                        if k == 'models' or k.startswith('models.'):
                            del sys.modules[k]
            return pr
    return None

class YOLOv5ModelWrapper:
    def __init__(self, model_path, device=""):
        self.model_path = model_path
        self.task = "detect"
        
        ensure_yolov5_on_path()
                
        import torch
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        
        dev_str = device if (device and device != "auto" and device != "自动选择") else ""
        self.device = select_device(dev_str)
        self.model = DetectMultiBackend(model_path, device=self.device)
        if hasattr(self.model, 'stride'):
            s = self.model.stride
            self.stride = int(max(s)) if hasattr(s, '__iter__') else int(s)
        else:
            self.stride = 32
        self.names = self.model.names if hasattr(self.model, 'names') else {}

    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None, verbose=False, save=False, project=None, name="predict", exist_ok=True, **kwargs):
        import cv2
        import numpy as np
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes, check_img_size
        from utils.plots import Annotator, colors
        
        img_files = []
        if os.path.isfile(source):
            img_files = [source]
        elif os.path.isdir(source):
            for root, _, files in os.walk(source):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')):
                        img_files.append(os.path.join(root, f))
        
        results = []
        s_imgsz = check_img_size(imgsz, s=self.stride)
        
        for path in img_files:
            t1 = time.time()
            im0 = cv2.imread(path)
            if im0 is None:
                continue
            orig_shape = im0.shape[:2]
            
            auto_letterbox = getattr(self.model, 'pt', False)
            im = letterbox(im0, s_imgsz, stride=self.stride, auto=auto_letterbox)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            im_tensor = torch.from_numpy(im).to(self.device).float() / 255.0
            im_tensor = im_tensor[None]
            
            t2 = time.time()
            pred = self.model(im_tensor)
            t3 = time.time()
            
            det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=1000)[0]
            t4 = time.time()
            
            xyxy_list = []
            conf_list = []
            cls_list = []
            
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], orig_shape).round()
                for item in det:
                    xyxy_list.append([float(x.item()) for x in item[:4]])
                    conf_list.append(float(item[4].item()))
                    cls_list.append(int(item[5].item()))
                    
            speed = {
                'preprocess': round((t2 - t1) * 1000, 2),
                'inference': round((t3 - t2) * 1000, 2),
                'postprocess': round((t4 - t3) * 1000, 2)
            }
            
            if save and project:
                save_out_dir = os.path.join(project, name)
                os.makedirs(save_out_dir, exist_ok=True)
                save_img_path = os.path.join(save_out_dir, os.path.basename(path))
                annotator = Annotator(im0.copy(), line_width=2, example=str(self.names))
                for b, score, c in zip(xyxy_list, conf_list, cls_list):
                    c_name = self.names.get(c, str(c))
                    annotator.box_label(b, f"{c_name} {score:.2f}", color=colors(c, True))
                cv2.imwrite(save_img_path, annotator.result())
                
            res_obj = SimpleYOLOv5Result(path, orig_shape, xyxy_list, conf_list, cls_list, self.names, speed)
            results.append(res_obj)
            
        return results

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

def is_yolov5_pt_model(model_path):
    if not model_path or not str(model_path).lower().endswith(".pt") or not os.path.isfile(model_path):
        return False
    fn = os.path.basename(model_path).lower()
    if fn.startswith("yolov5") or "yolov5" in fn:
        return True

    parent = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(parent).lower() == 'weights':
        exp_dir = os.path.dirname(parent)
        opt_yaml = os.path.join(exp_dir, 'opt.yaml')
        if os.path.isfile(opt_yaml):
            try:
                import yaml
                with open(opt_yaml, 'r', encoding='utf-8', errors='ignore') as f:
                    y = yaml.safe_load(f)
                if isinstance(y, dict):
                    w_str = str(y.get('weights', '')).lower()
                    if 'yolov5' in w_str:
                        return True
            except Exception:
                pass
    return False

def is_yolov5_onnx_model(model_path):
    if not str(model_path).lower().endswith(".onnx"):
        return False
    fn = os.path.basename(model_path).lower()
    if 'yolov5' in fn or fn.startswith('yolo5'):
        return True
    try:
        import onnx
        m = onnx.load(model_path, load_external_data=False)
        for prop in m.metadata_props:
            if prop.key.lower() == 'author' and 'ultralytics' in prop.value.lower():
                return False
            if prop.key.lower() == 'description' and 'ultralytics' in prop.value.lower():
                return False
        for out in m.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if len(dims) == 3 and dims[1] > dims[2] and dims[2] >= 5:
                return True
    except Exception:
        pass
    return False

def load_yolo_model(model_path, device=""):
    fn = os.path.basename(model_path).lower()
    if is_vajra_model(model_path):
        from vajra import Vajra
        return Vajra(model_path)

    if is_yolov5_pt_model(model_path) or is_yolov5_onnx_model(model_path):
        try:
            return YOLOv5ModelWrapper(model_path, device=device)
        except Exception as ex:
            print(f"[Info] YOLOv5 wrapper init note: {ex}, trying fallback...", flush=True)
            
    try:
        return YOLO(model_path)
    except Exception as ex:
        err_msg = str(ex).lower()
        if "vajra" in err_msg:
            from vajra import Vajra
            return Vajra(model_path)
        if ("yolov5" in err_msg or "models" in err_msg or "301" in err_msg or 
            "keyerror" in err_msg or "forwards compatible" in err_msg or "not forwards" in err_msg):
            return YOLOv5ModelWrapper(model_path, device=device)
        raise

def get_model_metadata(model):
    task = getattr(model, 'task', None)
    if not task and hasattr(model, 'model'):
        task = getattr(model.model, 'task', None)
    if not task:
        fn = os.path.basename(getattr(model, 'model_path', '') or str(getattr(model, 'ckpt_path', '')) or '').lower()
        if 'cls' in fn or 'classify' in fn:
            task = 'classify'
        elif 'seg' in fn or 'segment' in fn:
            task = 'segment'
        elif 'pose' in fn:
            task = 'pose'
        elif 'obb' in fn:
            task = 'obb'
        else:
            task = 'detect'

    native_imgsz = 640
    if hasattr(model, 'overrides') and isinstance(model.overrides, dict) and 'imgsz' in model.overrides:
        val = model.overrides['imgsz']
        if isinstance(val, (int, float)):
            native_imgsz = int(val)
        elif isinstance(val, (list, tuple)) and len(val) > 0:
            native_imgsz = int(val[0])
    elif hasattr(model, 'model') and hasattr(model.model, 'args') and isinstance(model.model.args, dict):
        args_d = model.model.args
        if 'imgsz' in args_d:
            val = args_d['imgsz']
            native_imgsz = int(val[0]) if isinstance(val, (list, tuple)) and len(val) > 0 else int(val)
        elif 'img_size' in args_d:
            val = args_d['img_size']
            native_imgsz = int(val[0]) if isinstance(val, (list, tuple)) and len(val) > 0 else int(val)
    elif task == 'classify':
        native_imgsz = 224

    raw_names = getattr(model, 'names', None)
    if callable(raw_names):
        try:
            raw_names = raw_names()
        except Exception:
            raw_names = None
    if not raw_names and hasattr(model, 'model'):
        raw_names = getattr(model.model, 'names', None)
        if callable(raw_names):
            try:
                raw_names = raw_names()
            except Exception:
                raw_names = None

    names = {}
    if isinstance(raw_names, dict):
        names = {int(k) if str(k).isdigit() else str(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {i: str(n) for i, n in enumerate(raw_names)}

    return {
        "task": task,
        "imgsz": native_imgsz,
        "nc": len(names),
        "names": names
    }

def process_source(model, source_path, conf=0.25, iou=0.45, imgsz=640, device=None, save_dir=None, save_img=False):
    # Ensure source exists
    if not os.path.exists(source_path):
        err_res = {
            "status": "error",
            "message": f"Source path not found: {source_path}",
            "image": source_path
        }
        print(f"__RESULT_JSON_START__{json.dumps(err_res, ensure_ascii=False)}__RESULT_JSON_END__", flush=True)
        return

    # Run inference
    kwargs = {
        "source": source_path,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False
    }
    if device is not None and device != "" and device != "auto":
        kwargs["device"] = device

    if save_img and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        kwargs["save"] = True
        kwargs["project"] = save_dir
        kwargs["name"] = "predict"
        kwargs["exist_ok"] = True

    try:
        results = model.predict(**kwargs)
    except Exception as ex:
        err_res = {
            "status": "error",
            "message": str(ex),
            "image": source_path
        }
        print(f"__RESULT_JSON_START__{json.dumps(err_res, ensure_ascii=False)}__RESULT_JSON_END__", flush=True)
        return

    for res in results:
        orig_shape = res.orig_shape # (h, w)
        img_h, img_w = orig_shape[0], orig_shape[1]
        img_path = res.path

        speed = {
            "preprocess": round(res.speed.get('preprocess', 0.0), 2),
            "inference": round(res.speed.get('inference', 0.0), 2),
            "postprocess": round(res.speed.get('postprocess', 0.0), 2)
        }

        detections = []
        names = res.names or {}

        # ===== OBB Results (Oriented Bounding Boxes) =====
        if res.obb is not None and len(res.obb) > 0:
            obb = res.obb
            for i in range(len(obb)):
                cls_id = int(obb.cls[i].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf_val = float(obb.conf[i].item())

                # xyxyxyxy: 4 corner points in pixel coords, shape (4, 2)
                corners = obb.xyxyxyxy[i].tolist()  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

                # xywhr: [cx, cy, w, h, rotation_rad]
                xywhr = obb.xywhr[i].tolist()
                cx, cy, bw, bh, rot_rad = xywhr

                # Compute axis-aligned bounding box from corners for compatibility
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                xmin = min(xs)
                ymin = min(ys)
                xmax = max(xs)
                ymax = max(ys)

                # Normalized center coords
                norm_cx = cx / img_w
                norm_cy = cy / img_h
                norm_w = bw / img_w
                norm_h = bh / img_h

                import math
                angle_deg = math.degrees(rot_rad)

                # Normalized corner points for OBB label saving
                norm_corners = [[round(p[0] / img_w, 6), round(p[1] / img_h, 6)] for p in corners]

                det_item = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    "norm_bbox": [round(norm_cx, 6), round(norm_cy, 6), round(norm_w, 6), round(norm_h, 6)],
                    "is_obb": True,
                    "angle": round(angle_deg, 2),
                    "corners": [[round(p[0], 2), round(p[1], 2)] for p in corners],
                    "norm_corners": norm_corners
                }
                detections.append(det_item)

        # ===== Standard Box Results (Detect / Segment) =====
        elif res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist() # [xmin, ymin, xmax, ymax]
                xywhn = boxes.xywhn[i].tolist() # [cx, cy, w, h] normalized

                det_item = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(x, 2) for x in xyxy],
                    "norm_bbox": [round(x, 6) for x in xywhn]
                }

                if res.masks is not None and len(res.masks) > i:
                    try:
                        seg_pts = res.masks.xy[i].tolist()
                        if len(seg_pts) >= 3:
                            det_item["is_segment"] = True
                            det_item["polygon"] = seg_pts
                            # Normalized polygon points
                            norm_poly = [[round(p[0] / img_w, 6), round(p[1] / img_h, 6)] for p in seg_pts]
                            det_item["norm_polygon"] = norm_poly
                    except Exception:
                        pass

                if res.keypoints is not None and len(res.keypoints) > i:
                    try:
                        kpts = res.keypoints.data[i].tolist()
                        if len(kpts) > 0:
                            det_item["is_pose"] = True
                            det_item["keypoints"] = kpts
                    except Exception:
                        pass

                detections.append(det_item)

        # ===== Image Classification Results =====
        elif res.probs is not None:
            top1_id = int(res.probs.top1)
            top1_name = names.get(top1_id, str(top1_id))
            top1_conf = float(res.probs.top1conf.item()) if hasattr(res.probs.top1conf, 'item') else float(res.probs.top1conf)

            top5_list = []
            if hasattr(res.probs, 'top5') and hasattr(res.probs, 'top5conf'):
                for c_id, c_conf in zip(res.probs.top5, res.probs.top5conf):
                    cid = int(c_id)
                    cval = float(c_conf.item()) if hasattr(c_conf, 'item') else float(c_conf)
                    top5_list.append({
                        "class_id": cid,
                        "class_name": names.get(cid, str(cid)),
                        "confidence": round(cval, 4)
                    })

            det_item = {
                "class_id": top1_id,
                "class_name": top1_name,
                "confidence": round(top1_conf, 4),
                "bbox": [0, 0, img_w, img_h],
                "norm_bbox": [0.5, 0.5, 1.0, 1.0],
                "is_classification": True,
                "top5": top5_list
            }
            detections.append(det_item)

        output_payload = {
            "status": "success",
            "image": img_path,
            "filename": os.path.basename(img_path),
            "width": img_w,
            "height": img_h,
            "speed": speed,
            "total_count": len(detections),
            "detections": detections
        }

        if save_img and save_dir:
            saved_img_path = os.path.join(save_dir, "predict", os.path.basename(img_path))
            output_payload["saved_image_path"] = saved_img_path

        print(f"__RESULT_JSON_START__{json.dumps(output_payload, ensure_ascii=False)}__RESULT_JSON_END__", flush=True)

def main():
    parser = argparse.ArgumentParser(description="NovaVision Inference & Auto-Labeling Bridge")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file (.pt)")
    parser.add_argument("--source", type=str, default="", help="Single image or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc.")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save predicted images")
    parser.add_argument("--save_img", action="store_true", help="Save annotated images")
    parser.add_argument("--interactive", action="store_true", help="Interactive stdin mode for continuous inference")
    parser.add_argument("--info_only", action="store_true", help="Only load model and output metadata JSON, then exit")

    args, _ = parser.parse_known_args()

    # For ONNX models, if device is not explicitly specified or is auto, use 'cpu' for stable ONNXRuntime execution
    if args.model.lower().endswith(".onnx") and (not args.device or args.device in ("", "auto", "自动选择")):
        args.device = "cpu"

    # Load model
    print(f"Loading model: {args.model}...", flush=True)
    try:
        model = load_yolo_model(args.model, device=args.device)
        meta = get_model_metadata(model)
        print(f"__MODEL_LOADED__{json.dumps(meta, ensure_ascii=False)}", flush=True)
        if args.info_only:
            return
    except Exception as ex:
        print(f"__MODEL_ERROR__{str(ex)}", flush=True)
        sys.exit(1)

    if args.interactive:
        # Interactive mode: read lines from stdin
        # Format: source_path|conf|iou|imgsz|device|save_dir|save_img
        for line in sys.stdin:
            line = line.strip()
            if not line or line.lower() == "exit" or line.lower() == "quit":
                break
            parts = line.split("|")
            src = parts[0]
            conf = float(parts[1]) if len(parts) > 1 and parts[1] else args.conf
            iou = float(parts[2]) if len(parts) > 2 and parts[2] else args.iou
            imgsz = int(parts[3]) if len(parts) > 3 and parts[3] else args.imgsz
            dev = parts[4] if len(parts) > 4 and parts[4] else args.device
            sdir = parts[5] if len(parts) > 5 and parts[5] else args.save_dir
            simg = (parts[6].lower() in ['1', 'true', 'yes']) if len(parts) > 6 else args.save_img

            process_source(model, src, conf=conf, iou=iou, imgsz=imgsz, device=dev, save_dir=sdir, save_img=simg)
    else:
        if args.source:
            process_source(model, args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                           device=args.device, save_dir=args.save_dir, save_img=args.save_img)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
