# -*- coding: utf-8 -*-
"""
NovaVision - YOLO Model Evaluation / Validation Bridge
Runs model.val() and streams real metrics back to C# via JSON markers.
"""
import os
import sys

# Headless matplotlib Agg backend configuration
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass

import argparse
import json
from ultralytics import YOLO


def str2bool(v):
    if isinstance(v, bool):
        return v
    v_lower = str(v).lower()
    if v_lower in ('yes', 'true', 't', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', '0'):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="NovaVision YOLO Evaluation Bridge")
    parser.add_argument('--model', type=str, required=True, help='Model .pt file path')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None, help='Device (0, cpu, etc.)')
    parser.add_argument('--split', type=str, default='val', help='Dataset split: val, test, train')
    parser.add_argument('--plots', type=str2bool, default=True, help='Generate PR curves and confusion matrix')

    args = parser.parse_args()

    print(f"[评估] 正在加载模型: {args.model}", flush=True)
    sys.stdout.flush()

    if not os.path.exists(args.model):
        print(f"__EVAL_ERROR__模型文件不存在: {args.model}", flush=True)
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"__EVAL_ERROR__数据集配置文件不存在: {args.data}", flush=True)
        sys.exit(1)

    model = YOLO(args.model)

    print(f"[评估] 模型加载完毕，开始验证评估...", flush=True)
    print(f"[评估] 数据集: {args.data}", flush=True)
    print(f"[评估] 参数: batch={args.batch}, imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}", flush=True)
    sys.stdout.flush()

    val_kwargs = {
        'data': args.data,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'split': args.split,
        'verbose': True,
        'plots': args.plots,
    }

    if args.device and args.device != 'auto':
        val_kwargs['device'] = args.device

    try:
        results = model.val(**val_kwargs)

        # Extract metrics from results
        map50 = 0.0
        map50_95 = 0.0
        precision = 0.0
        recall = 0.0
        fitness = 0.0
        speed_preprocess = 0.0
        speed_inference = 0.0
        speed_postprocess = 0.0

        if hasattr(results, 'obb') and results.obb is not None:
            metric_obj = results.obb
            map50 = float(getattr(metric_obj, 'map50', 0.0))
            map50_95 = float(getattr(metric_obj, 'map', 0.0))
            precision = float(getattr(metric_obj, 'mp', 0.0))
            recall = float(getattr(metric_obj, 'mr', 0.0))
        elif hasattr(results, 'seg') and results.seg is not None:
            metric_obj = results.seg
            map50 = float(getattr(metric_obj, 'map50', 0.0))
            map50_95 = float(getattr(metric_obj, 'map', 0.0))
            precision = float(getattr(metric_obj, 'mp', 0.0))
            recall = float(getattr(metric_obj, 'mr', 0.0))
        elif hasattr(results, 'pose') and results.pose is not None:
            metric_obj = results.pose
            map50 = float(getattr(metric_obj, 'map50', 0.0))
            map50_95 = float(getattr(metric_obj, 'map', 0.0))
            precision = float(getattr(metric_obj, 'mp', 0.0))
            recall = float(getattr(metric_obj, 'mr', 0.0))
        elif hasattr(results, 'box') and results.box is not None:
            metric_obj = results.box
            map50 = float(getattr(metric_obj, 'map50', 0.0))
            map50_95 = float(getattr(metric_obj, 'map', 0.0))
            precision = float(getattr(metric_obj, 'mp', 0.0))
            recall = float(getattr(metric_obj, 'mr', 0.0))
        elif hasattr(results, 'top1'):
            precision = float(getattr(results, 'top1', 0.0))
            recall = float(getattr(results, 'top5', 0.0))
            map50 = precision
            map50_95 = recall
        elif hasattr(results, 'results_dict'):
            rd = results.results_dict
            map50 = float(rd.get('metrics/mAP50(B)', rd.get('metrics/accuracy_top1', rd.get('metrics/mAP_0.5', 0.0))))
            map50_95 = float(rd.get('metrics/mAP50-95(B)', rd.get('metrics/accuracy_top5', rd.get('metrics/mAP_0.5:0.95', 0.0))))
            precision = float(rd.get('metrics/precision(B)', rd.get('metrics/accuracy_top1', 0.0)))
            recall = float(rd.get('metrics/recall(B)', rd.get('metrics/accuracy_top5', 0.0)))

        if hasattr(results, 'fitness'):
            fitness = float(results.fitness)

        if hasattr(results, 'speed'):
            speed = results.speed
            speed_preprocess = float(speed.get('preprocess', 0.0))
            speed_inference = float(speed.get('inference', 0.0))
            speed_postprocess = float(speed.get('postprocess', 0.0))

        # Per-class metrics
        per_class = []
        if hasattr(results, 'box') and hasattr(results.box, 'ap50'):
            ap50_arr = results.box.ap50
            ap_arr = results.box.ap  # mAP50-95 per class
            p_arr = results.box.p if hasattr(results.box, 'p') else None
            r_arr = results.box.r if hasattr(results.box, 'r') else None
            names = results.names if hasattr(results, 'names') else {}

            if hasattr(ap50_arr, '__len__'):
                for i in range(len(ap50_arr)):
                    cls_name = names.get(i, f"class_{i}") if isinstance(names, dict) else str(i)
                    entry = {
                        "class_id": i,
                        "class_name": cls_name,
                        "ap50": round(float(ap50_arr[i]), 4),
                        "ap50_95": round(float(ap_arr[i]), 4) if ap_arr is not None and i < len(ap_arr) else 0.0,
                    }
                    if p_arr is not None and i < len(p_arr):
                        entry["precision"] = round(float(p_arr[i]), 4)
                    if r_arr is not None and i < len(r_arr):
                        entry["recall"] = round(float(r_arr[i]), 4)
                    per_class.append(entry)

        # Save directory for plots
        save_dir = ""
        if hasattr(results, 'save_dir'):
            save_dir = str(results.save_dir)

        payload = {
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fitness": round(fitness, 4),
            "speed_preprocess": round(speed_preprocess, 2),
            "speed_inference": round(speed_inference, 2),
            "speed_postprocess": round(speed_postprocess, 2),
            "per_class": per_class,
            "save_dir": save_dir,
        }

        print(f"__EVAL_RESULT__{json.dumps(payload, ensure_ascii=False)}", flush=True)
        sys.stdout.flush()

    except Exception as ex:
        print(f"__EVAL_ERROR__{str(ex)}", flush=True)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()
