# -*- coding: utf-8 -*-
"""
NovaVision - Industrial YOLO FastAPI Inference Server
Supports REST API endpoints for high-throughput object detection and Swagger UI.
"""
import sys
import os
import io
import time
import argparse
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="NovaVision YOLO Inference Microservice")
parser.add_argument('--model', type=str, required=True, help='Path to YOLO model (.pt, .onnx, .engine)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
parser.add_argument('--port', type=int, default=8000, help='Port to bind')
parser.add_argument('--conf', type=float, default=0.25, help='Default confidence threshold')
parser.add_argument('--iou', type=float, default=0.45, help='Default NMS IOU threshold')
parser.add_argument('--device', type=str, default=None, help='Device (e.g. 0, cpu)')

args = parser.parse_args()

print(f"[Server] Loading YOLO model from: {args.model} ...", flush=True)
sys.stdout.flush()

try:
    from bridge_predict import load_yolo_model
    model = load_yolo_model(args.model, device=args.device or "")
    print(f"[Server] Model loaded successfully! Task: {getattr(model, 'task', 'detect')}", flush=True)
except Exception as e:
    print(f"[Server Error] Failed to load model: {e}", flush=True)
    sys.exit(1)

app = FastAPI(
    title="YOLO Platform - 工业视觉推理服务",
    description="支持多种 YOLO 任务的高性能 Web 推理服务 API，提供目标检测、实例分割、分类等标准化 HTTP 接口。",
    version="2.6.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "YOLO Platform Industrial Inference Server",
        "model": os.path.basename(args.model),
        "docs_url": f"http://localhost:{args.port}/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "time": time.time()}

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="待检测图像文件 (JPG/PNG/BMP)"),
    conf: float = args.conf,
    iou: float = args.iou
):
    start_t = time.time()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "无法解码图像，请上传有效的图片文件。"})
        
        results = model.predict(img, conf=conf, iou=iou, device=args.device, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item())
                    cname = res.names.get(cls_id, f"class_{cls_id}")
                    score = float(b.conf[0].item())
                    box = [round(float(x), 2) for x in b.xyxy[0].tolist()]
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cname,
                        "confidence": round(score, 4),
                        "bbox": {
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "width": round(box[2] - box[0], 2),
                            "height": round(box[3] - box[1], 2)
                        }
                    })
        
        elapsed_ms = round((time.time() - start_t) * 1000, 2)
        print(f"[API] POST /predict -> 成功检测到 {len(detections)} 个目标, 耗时 {elapsed_ms} ms", flush=True)
        sys.stdout.flush()

        return {
            "code": 0,
            "message": "success",
            "image_filename": file.filename,
            "inference_time_ms": elapsed_ms,
            "count": len(detections),
            "detections": detections
        }
    except Exception as ex:
        print(f"[API Error] /predict: {ex}", flush=True)
        sys.stdout.flush()
        return JSONResponse(status_code=500, content={"error": str(ex)})

@app.post("/predict_annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float = args.conf,
    iou: float = args.iou
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})
        
        results = model.predict(img, conf=conf, iou=iou, device=args.device, verbose=False)
        if results and len(results) > 0:
            annotated_img = results[0].plot()
            _, encoded_img = cv2.imencode('.jpg', annotated_img)
            return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        return Response(content=contents, media_type="image/jpeg")
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})

if __name__ == '__main__':
    print(f"==================================================", flush=True)
    print(f"  NovaVision FastAPI 推理微服务已就绪！", flush=True)
    print(f"  * 本地访问地址: http://127.0.0.1:{args.port}", flush=True)
    print(f"  * 交互式文档:   http://127.0.0.1:{args.port}/docs", flush=True)
    print(f"==================================================", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
