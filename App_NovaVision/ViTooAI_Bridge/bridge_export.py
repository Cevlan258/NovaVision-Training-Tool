# -*- coding: utf-8 -*-
"""
NovaVision - Unified Model Export Bridge
Supports Ultralytics (YOLO8/11/12/26) and YOLOv5 models.
Formats: ONNX, TensorRT (Engine), TorchScript, OpenVINO, CoreML, TFLite.
"""
import os
import sys
import types
import faulthandler

try:
    faulthandler.enable()
except Exception:
    pass

sys.modules['albumentations'] = None
sys.modules.setdefault('torchvision.models', types.ModuleType('torchvision.models'))

os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import zipfile
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    vl = str(v).lower()
    return vl in ('yes', 'true', 't', '1')

from bridge_common import configure_windows_dll_path

configure_windows_dll_path()

import torch

def ensure_yolov5_on_path():
    bridge_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(bridge_dir)
    possible_roots = [
        os.path.join(root_dir, 'yolov5'),
        os.path.join(bridge_dir, 'yolov5'),
        r'F:\Yolo26\Debug\yolov5',
        r'F:\Yolo26\Debug\App_NovaVision\yolov5',
        r'F:\YOLO_ training_tool\yolo\model\yolov5'
    ]
    for pr in reversed(possible_roots):
        if os.path.isdir(pr) and os.path.isfile(os.path.join(pr, 'export.py')):
            if pr in sys.path:
                sys.path.remove(pr)
            sys.path.insert(0, pr)
            return pr
    return None

def is_yolov5_model(model_path):
    if not model_path or not os.path.isfile(model_path):
        return False
    fn = os.path.basename(model_path).lower()
    if fn.startswith('yolov5') or 'yolov5' in fn:
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
    if not model_path or not os.path.isfile(model_path):
        return False
    fn = os.path.basename(model_path).lower()
    if 'vajra' in fn:
        return True
    if str(model_path).endswith('.pt'):
        try:
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

def inspect_and_log_onnx(onnx_path, arch_desc=""):
    if not os.path.isfile(onnx_path):
        return
    try:
        import onnx
        from onnx import TensorProto

        dtype_map = {
            TensorProto.FLOAT: 'float32',
            TensorProto.FLOAT16: 'float16',
            TensorProto.DOUBLE: 'float64',
            TensorProto.INT32: 'int32',
            TensorProto.INT64: 'int64',
            TensorProto.UINT8: 'uint8',
            TensorProto.INT8: 'int8',
            TensorProto.BOOL: 'bool'
        }

        m = onnx.load(onnx_path)
        print("\n" + "=" * 65, flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] 🔍 [ONNX结构校验] 成功加载导出的 ONNX 模型:", flush=True)
        print(f"  • 模型文件: {onnx_path} ({os.path.getsize(onnx_path) / (1024*1024):.2f} MB)", flush=True)

        for i, inp in enumerate(m.graph.input):
            dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or '?') for d in inp.type.tensor_type.shape.dim]
            elem_type = inp.type.tensor_type.elem_type
            dtype_name = dtype_map.get(elem_type, f'type_{elem_type}')
            print(f"  • 输入张量 #{i}: name='{inp.name}', shape={dims}, dtype={dtype_name}", flush=True)

        # 统计核心权重精度与量化特征
        init_dtypes = [init.data_type for init in m.graph.initializer]
        if init_dtypes:
            fp16_cnt = sum(1 for dt in init_dtypes if dt == TensorProto.FLOAT16)
            fp32_cnt = sum(1 for dt in init_dtypes if dt == TensorProto.FLOAT)
            tot_init = len(init_dtypes)
            if fp16_cnt / tot_init >= 0.7:
                prec_info = f"FP16 半精度浮点 ({fp16_cnt}/{tot_init} 个权重层，体积缩减 ~50%)"
                print(f"  • 核心权重精度: {prec_info}", flush=True)
                print(f"  • 接口标准规范: 对标标准参考模型规格 (输入输出保持 float32 即插即用，核心计算全链路 FP16 硬件加速)", flush=True)
            elif fp32_cnt / tot_init >= 0.7:
                prec_info = f"FP32 单精度浮点 ({fp32_cnt}/{tot_init} 个权重层)"
                print(f"  • 核心权重精度: {prec_info}", flush=True)
            else:
                prec_info = f"混合精度 (FP16: {fp16_cnt}, FP32: {fp32_cnt})"
                print(f"  • 核心权重精度: {prec_info}", flush=True)

        onnx_arch_feature = ""
        for i, out in enumerate(m.graph.output):
            dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or '?') for d in out.type.tensor_type.shape.dim]
            elem_type = out.type.tensor_type.elem_type
            dtype_name = dtype_map.get(elem_type, f'type_{elem_type}')

            detail = ""
            if len(dims) == 2:
                detail = f" (图像分类概率输出: [batch, {dims[1]}])"
                onnx_arch_feature = f"YOLO 图像分类架构 (类别数: {dims[1]})"
            elif len(dims) == 4 and (dims[1] == 32 or dims[1] == 1):
                detail = f" (实例分割掩码原型: [batch, {dims[1]}, {dims[2]}, {dims[3]}])"
            elif len(dims) == 3 and dims[1] == 300 and dims[2] == 6:
                detail = " (YOLO26 原生端到端目标检测格式: [batch, 300, 6])"
                onnx_arch_feature = "YOLO26 原生端到端无NMS目标检测架构 (直接输出 Top 300 候选框，免后处理 NMS)"
            elif len(dims) == 3 and dims[1] == 300 and dims[2] > 6:
                detail = f" (YOLO26 端到端任务输出: [batch, 300, {dims[2]}])"
                onnx_arch_feature = f"YOLO26 端到端任务架构 ([batch, 300, {dims[2]}])"
            elif len(dims) == 3 and (dims[2] == 8400 or (isinstance(dims[2], int) and dims[2] > 1000 and dims[1] < dims[2])):
                detail = f" (YOLOv8-11-12 密集预测格式: [batch, {dims[1]}, {dims[2]}])"
                onnx_arch_feature = f"YOLOv8 / YOLO11 / YOLO12 密集预测架构 (输出: [batch, {dims[1]}, {dims[2]}])"
            elif len(dims) == 3 and dims[1] == 25200:
                detail = f" (YOLOv5 经典多尺度锚框格式: [batch, 25200, {dims[2]}])"
                onnx_arch_feature = f"YOLOv5 经典多尺度锚框架构 (输出: [batch, 25200, {dims[2]}])"

            print(f"  • 输出张量 #{i}: name='{out.name}', shape={dims}, dtype={dtype_name}{detail}", flush=True)

        if onnx_arch_feature:
            print(f"  • 架构特征: {onnx_arch_feature}", flush=True)
        elif arch_desc:
            print(f"  • 架构特征: {arch_desc}", flush=True)
        print("=" * 65 + "\n", flush=True)
    except Exception as ex:
        print(f"[ONNX结构校验提示] 未能解析 ONNX 结构: {ex}", flush=True)

def inspect_pt_file_directly(path):
    res = {'arch': 'YOLO', 'task': 'det', 'is_yolo26': False, 'is_v5': False}
    if not os.path.isfile(path):
        return res
    raw = b''
    try:
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            pkl_name = next((n for n in names if n.endswith('data.pkl')), None)
            if pkl_name:
                raw = z.read(pkl_name)
    except Exception:
        pass

    # 1. 任务类型判定 (从权重内保存的 PyTorch 模型架构模块名直接提取)
    if b'PoseModel' in raw or b'Pose26' in raw or b'modules.head\nPose' in raw or b'.head\nPose' in raw:
        res['task'] = 'pose'
    elif b'SegmentationModel' in raw or b'Segment26' in raw or b'modules.head\nSegment' in raw or b'.head\nSegment' in raw:
        res['task'] = 'seg'
    elif b'OBBModel' in raw or b'OBB26' in raw or b'modules.head\nOBB' in raw or b'.head\nOBB' in raw:
        res['task'] = 'obb'
    elif b'ClassificationModel' in raw or b'modules.head\nClassify' in raw or b'.head\nClassify' in raw:
        res['task'] = 'cls'
    elif b'DetectionModel' in raw or b'Detect26' in raw or b'modules.head\nDetect' in raw or b'.head\nDetect' in raw:
        res['task'] = 'det'
    else:
        fn_lower = os.path.basename(path).lower()
        if any(k in fn_lower for k in ['-seg', '_seg', '.seg', 'seg.']): res['task'] = 'seg'
        elif any(k in fn_lower for k in ['-pose', '_pose', '.pose', 'pose.']): res['task'] = 'pose'
        elif any(k in fn_lower for k in ['-obb', '_obb', '.obb', 'obb.']): res['task'] = 'obb'
        elif any(k in fn_lower for k in ['-cls', '_cls', '.cls', 'cls.']): res['task'] = 'cls'
        else: res['task'] = 'det'

    # 2. 架构判定 (根据权重内部 Head 模块、one2one 标记及 YAML 配置判定)
    if (b'Pose26' in raw or b'Segment26' in raw or b'Detect26' in raw or b'OBB26' in raw or 
        b'one2one_cv2' in raw or b'one2one_cv3' in raw or b'yolo26' in raw):
        res['arch'] = 'YOLO26'
        res['is_yolo26'] = True
    elif any(k in raw for k in [b'vajra.nn', b'vajra.models', b'vajra.core', b'Merudanda', b'VajraV1', b'VajraV2']):
        res['arch'] = 'VajraV1'
        res['is_vajra'] = True
    elif b'yolov12' in raw or b'yolo12' in raw or b'AAttn' in raw:
        res['arch'] = 'YOLOv12'
    elif b'yolo11' in raw or b'C3k2' in raw:
        res['arch'] = 'YOLO11'
    elif b'yolov8' in raw or b'yolo8' in raw:
        res['arch'] = 'YOLOv8'
    elif b'yolov5' in raw or b'yolo5' in raw or b'models.yolo' in raw:
        res['arch'] = 'YOLOv5'
        res['is_v5'] = True
    else:
        fn_lower = os.path.basename(path).lower()
        if '26' in fn_lower:
            res['arch'] = 'YOLO26'
            res['is_yolo26'] = True
        elif 'vajra' in fn_lower:
            res['arch'] = 'VajraV1'
            res['is_vajra'] = True
        elif '12' in fn_lower:
            res['arch'] = 'YOLOv12'
        elif '11' in fn_lower:
            res['arch'] = 'YOLO11'
        elif 'yolov8' in fn_lower or 'yolo8' in fn_lower:
            res['arch'] = 'YOLOv8'
        elif 'yolov5' in fn_lower or 'yolo5' in fn_lower:
            res['arch'] = 'YOLOv5'
            res['is_v5'] = True

    return res

def main():
    parser = argparse.ArgumentParser(description="NovaVision Unified Model Export Bridge")
    parser.add_argument('--model', type=str, required=True, help='Path to source .pt model')
    parser.add_argument('--format', type=str, default='onnx', help='Target format: onnx, engine, torchscript, openvino, etc.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution size')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--half', type=str2bool, default=False, help='FP16 half-precision export')
    parser.add_argument('--dynamic', type=str2bool, default=False, help='Dynamic axes')
    parser.add_argument('--device', type=str, default='', help='Device: 0 or cpu')

    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"[错误] 模型文件不存在: {model_path}", flush=True)
        sys.exit(1)

    fmt_raw = args.format.lower()
    if 'onnx' in fmt_raw:
        fmt = 'onnx'
    elif 'engine' in fmt_raw or 'tensorrt' in fmt_raw:
        fmt = 'engine'
    elif 'torchscript' in fmt_raw:
        fmt = 'torchscript'
    elif 'openvino' in fmt_raw:
        fmt = 'openvino'
    elif 'coreml' in fmt_raw:
        fmt = 'coreml'
    elif 'tflite' in fmt_raw:
        fmt = 'tflite'
    else:
        fmt = fmt_raw

    pt_info = inspect_pt_file_directly(model_path)
    is_v5 = pt_info['is_v5'] or is_yolov5_model(model_path)
    is_yolo26 = pt_info['is_yolo26']
    is_vajra = (pt_info.get('is_vajra', False) or is_vajra_model(model_path)) and not is_yolo26
    task = pt_info['task']
    is_yolo12 = (pt_info['arch'] == 'YOLOv12')
    is_yolo11 = (pt_info['arch'] == 'YOLO11')
    is_yolov8 = (pt_info['arch'] == 'YOLOv8')
    arch_prefix = pt_info['arch']

    arch_name = ""
    expected_output = ""
    if is_vajra:
        arch_name = "VajraV1 (VayuAI SOTA) 实时检测架构"
        if task == 'seg':
            expected_output = "output0 [1, 4+nc+32, 8400], output1 [1, 32, 160, 160] (8400候选点 + 掩码原型)"
        elif task == 'pose':
            expected_output = "output0 [1, 4+1+kpt*3, 8400] (8400 候选关键点输出)"
        elif task == 'obb':
            expected_output = "output0 [1, 4+nc+1, 8400] (8400 候选点输出，含旋转角度)"
        else:
            expected_output = "output0 [1, 4+nc, 8400] (8400 候选点输出，需转置 + NMS)"
    elif is_v5:
        arch_name = "YOLOv5 经典多尺度锚框 (Grid Anchors)"
        expected_output = "output0 [1, 25200, 5+nc] (多尺度 Anchor 解码 + NMS)"
    elif task == 'seg':
        if is_yolo26:
            arch_name = f"{arch_prefix} 原生端到端实例分割架构"
            expected_output = "output0 [1, 300, 4+nc+32], output1 [1, 32, 160, 160] (端到端输出 + 掩码原型)"
        else:
            arch_name = f"{arch_prefix} 实例分割密集预测架构"
            expected_output = "output0 [1, 4+nc+32, 8400], output1 [1, 32, 160, 160] (8400候选点 + 掩码原型)"
    elif task == 'pose':
        if is_yolo26:
            arch_name = f"{arch_prefix} 原生端到端姿态估计 / 关键点检测架构"
            expected_output = "output0 [1, 300, 4+1+kpt*3] (端到端直接输出 Top 300 关键点目标)"
        else:
            arch_name = f"{arch_prefix} 姿态估计密集预测架构"
            expected_output = "output0 [1, 4+1+kpt*3, 8400] (8400 候选关键点输出)"
    elif task == 'cls':
        arch_name = f"{arch_prefix} 图像分类预测架构"
        expected_output = "output0 [1, num_classes] (分类置信度分布)"
    elif task == 'obb':
        if is_yolo26:
            arch_name = f"{arch_prefix} 原生端到端旋转目标检测 (OBB)"
            expected_output = "output0 [1, 300, 4+nc+1] (端到端直接输出 Top 300 旋转候选框)"
        else:
            arch_name = f"{arch_prefix} 旋转目标检测密集预测架构 (OBB)"
            expected_output = "output0 [1, 4+nc+1, 8400] (8400 候选点输出，含旋转角度)"
    elif is_yolo26:
        arch_name = "YOLO26 原生端到端无NMS架构 (End-to-End NMS-Free)"
        expected_output = "output0 [1, 300, 6] (端到端直接输出 Top 300 候选框，免后处理 NMS)"
    else:
        arch_name = f"{arch_prefix} 无锚框密集预测架构 (Dense Anchor-Free)"
        expected_output = "output0 [1, 4+nc, 8400] (8400 候选点输出，需转置 + NMS)"

    print(f"[{time.strftime('%H:%M:%S')}] 开始模型导出工程...", flush=True)
    print(f"[配置] 模型: {model_path}", flush=True)
    print(f"[架构识别] {arch_name}", flush=True)
    print(f"[格式规格] 预期输出张量: {expected_output}", flush=True)
    print(f"[配置] 目标格式: {fmt.upper()}, 分辨率: {args.imgsz}, 批次: {args.batch}, FP16: {args.half}, Dynamic: {args.dynamic}", flush=True)

    dev = args.device if (args.device and args.device != 'auto' and args.device != '自动选择') else ''

    cuda_avail = False
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except Exception:
        pass

    # 智能硬件调度：半精度 FP16 模式优先调用 GPU 原生加速（避免 CPU 导出产生 Cast 转换包装）
    if not dev or dev.lower() in ('auto', '自动选择'):
        dev = '0' if cuda_avail else 'cpu'
    elif args.half and dev.lower() == 'cpu' and cuda_avail:
        print(f"[{time.strftime('%H:%M:%S')}] ⚡ 检测到本机具备 NVIDIA CUDA GPU，半精度 FP16 模式已自动升级为 GPU 原生硬件加速导出 (Device 0)！", flush=True)
        dev = '0'

    if is_v5:
        # Export using YOLOv5 export.py
        bridge_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(bridge_dir)
        possible_roots = [
            os.path.join(root_dir, 'yolov5'),
            os.path.join(bridge_dir, 'yolov5'),
            r'F:\YOLO_ training_tool\yolo\model\yolov5'
        ]
        for pr in reversed(possible_roots):
            if os.path.isdir(pr):
                if pr in sys.path:
                    sys.path.remove(pr)
                sys.path.insert(0, pr)

        try:
            from export import run as run_v5_export
            print(f"[{time.strftime('%H:%M:%S')}] 正在调用 YOLOv5 原生导出引擎...", flush=True)
            export_dev = dev if dev else ('0' if cuda_avail else 'cpu')
            if args.half and export_dev == 'cpu' and cuda_avail:
                export_dev = '0'
            run_v5_export(
                weights=model_path,
                imgsz=(args.imgsz, args.imgsz),
                batch_size=args.batch,
                device=export_dev,
                include=(fmt,),
                half=args.half,
                dynamic=args.dynamic,
                simplify=True
            )
            print(f"[{time.strftime('%H:%M:%S')}] 🎉 YOLOv5 模型导出完成！", flush=True)
        except Exception as ex:
            print(f"[错误] YOLOv5 导出失败: {ex}", flush=True)
            sys.exit(1)
    elif is_vajra:
        # Export using Vajra
        try:
            from vajra import Vajra
            print(f"[{time.strftime('%H:%M:%S')}] 正在调用 VajraV1 原生导出引擎...", flush=True)
            model = Vajra(model_path)
            export_kwargs = {
                'format': fmt,
                'img_size': args.imgsz,
                'batch': args.batch,
                'half': args.half,
                'dynamic': args.dynamic,
                'simplify': False
            }
            if dev:
                export_kwargs['device'] = dev
            model.export(**export_kwargs)
            print(f"[{time.strftime('%H:%M:%S')}] 🎉 VajraV1 模型导出完成！", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"[错误] VajraV1 导出失败: {ex}", flush=True)
            sys.exit(1)
    else:
        # Export using Ultralytics
        try:
            from ultralytics import YOLO
            print(f"[{time.strftime('%H:%M:%S')}] 正在调用 Ultralytics 导出引擎...", flush=True)
            model = YOLO(model_path)

            # 动态检查模型网络 Head 结构
            head = None
            if hasattr(model, 'model') and hasattr(model.model, 'model') and len(model.model.model) > 0:
                head = model.model.model[-1]

            # 若为 YOLO26 目标检测架构（包含 one2one 预测头分支且为纯 Detect 任务），确保恢复 end2end=True 输出 [1, 300, ...]
            is_head_yolo26_det = (head is not None and (hasattr(head, 'one2one_cv2') or '26' in head.__class__.__name__) and head.__class__.__name__ == 'Detect') or (is_yolo26 and getattr(model, 'task', None) == 'detect')
            if is_head_yolo26_det and head is not None:
                if hasattr(head, 'end2end'):
                    head.end2end = True
                if hasattr(model.model, 'end2end'):
                    model.model.end2end = True

            export_kwargs = {
                'format': fmt,
                'imgsz': args.imgsz,
                'batch': args.batch,
                'half': args.half,
                'dynamic': args.dynamic,
                'simplify': False
            }
            if is_head_yolo26_det:
                export_kwargs['nms'] = False
            if dev:
                export_kwargs['device'] = dev
            model.export(**export_kwargs)
            print(f"[{time.strftime('%H:%M:%S')}] 🎉 Ultralytics 模型导出完成！", flush=True)
        except Exception as ex:
            print(f"[错误] 导出失败: {ex}", flush=True)
            sys.exit(1)

    # 导出完成后，若为 ONNX 格式，自动检验并打印输入输出张量结构
    if fmt == 'onnx':
        onnx_candidates = [
            os.path.splitext(model_path)[0] + '.onnx',
            os.path.join(os.path.dirname(model_path), os.path.splitext(os.path.basename(model_path))[0] + '.onnx'),
            os.path.join(os.getcwd(), os.path.splitext(os.path.basename(model_path))[0] + '.onnx')
        ]
        target_onnx = None
        for cand in onnx_candidates:
            if os.path.isfile(cand):
                target_onnx = cand
                break

        if target_onnx and args.half:
            # 严格对标参考模型 (如 yolov8n.onnx) 标准工业规格：
            # 1. 核心卷积与网络权重压缩为 FP16 (float16)，体积减半 (~50%)，算力与显存翻倍优化
            # 2. 外部接口保留工业标准 float32 (含 Cast 包装)，确保 C# / C++ / OpenCV / ONNX Runtime 即插即用不报错
            try:
                import onnx
                from onnx import helper, TensorProto
                m_check = onnx.load(target_onnx)
                fp16_cnt = sum(1 for init in m_check.graph.initializer if init.data_type == TensorProto.FLOAT16)
                total_init = len(m_check.graph.initializer)
                
                # 若权重尚未量化为 FP16 (如 YOLOv5 或部分导出引擎未量化)，执行对齐参考模型的标准 FP16 转换
                if total_init > 0 and (fp16_cnt / total_init < 0.7):
                    print(f"[{time.strftime('%H:%M:%S')}] ⚡ 正在对标参考模型执行半精度量化 (FP32 兼容接口 + FP16 权重加速)...", flush=True)
                    from onnxruntime.transformers import float16
                    m_check = float16.convert_float_to_float16(m_check, keep_io_types=True)
                    onnx.save(m_check, target_onnx)
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ 成功生成对标参考模型规格的半精度 FP16 ONNX！", flush=True)
                else:
                    # 若接口本身是 float16 (例如 GPU 直接导出的模型)，封装标准 float32 输入输出 Cast 适配层，严格对齐参考模型
                    has_fp16_io = any(i.type.tensor_type.elem_type == TensorProto.FLOAT16 for i in m_check.graph.input) or \
                                  any(o.type.tensor_type.elem_type == TensorProto.FLOAT16 for o in m_check.graph.output)
                    if has_fp16_io:
                        print(f"[{time.strftime('%H:%M:%S')}] ⚡ 正在对标参考模型封装标准 FP32 兼容输入输出接口...", flush=True)
                        for inp in m_check.graph.input:
                            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                                orig_name = inp.name
                                cast_out = f'graph_input_cast_{orig_name}'
                                inp.type.tensor_type.elem_type = TensorProto.FLOAT
                                for node in m_check.graph.node:
                                    for idx, in_name in enumerate(node.input):
                                        if in_name == orig_name:
                                            node.input[idx] = cast_out
                                c_node = helper.make_node('Cast', inputs=[orig_name], outputs=[cast_out], name=f'graph_input_cast_{orig_name}', to=int(TensorProto.FLOAT16))
                                m_check.graph.node.insert(0, c_node)

                        for out in m_check.graph.output:
                            if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
                                orig_name = out.name
                                cast_in = f'graph_output_cast_{orig_name}'
                                out.type.tensor_type.elem_type = TensorProto.FLOAT
                                for node in m_check.graph.node:
                                    for idx, out_name in enumerate(node.output):
                                        if out_name == orig_name:
                                            node.output[idx] = cast_in
                                c_node = helper.make_node('Cast', inputs=[cast_in], outputs=[orig_name], name=f'graph_output_cast_{orig_name}', to=int(TensorProto.FLOAT))
                                m_check.graph.node.append(c_node)

                        onnx.save(m_check, target_onnx)
                        print(f"[{time.strftime('%H:%M:%S')}] ✅ 成功对齐参考模型接口规格 (FP32 I/O + FP16 Weights)！", flush=True)
            except Exception as e:
                print(f"[提示] 参考模型规格对齐跳过: {e}", flush=True)

        if target_onnx:
            inspect_and_log_onnx(target_onnx, arch_name)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)

if __name__ == '__main__':
    main()
