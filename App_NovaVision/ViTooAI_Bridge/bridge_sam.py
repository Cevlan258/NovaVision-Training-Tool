# -*- coding: utf-8 -*-
"""
NovaVision Studio - SAM & Intelligent Polygon Segmentation Engine
High-precision point prompt & bounding box shrink-wrap segmentation.
"""
import os
import sys
import argparse
import json
import cv2
import numpy as np

def segment_by_point(img, px, py):
    """
    Given an image and a normalized click point (px, py),
    extract a precise polygon contour enclosing the clicked target.
    """
    h, w = img.shape[:2]
    cx = int(np.clip(px * w, 0, w - 1))
    cy = int(np.clip(py * h, 0, h - 1))

    # Adaptive ROI around click point (up to 45% of image size)
    roi_w = max(60, int(w * 0.45))
    roi_h = max(60, int(h * 0.45))

    x1 = max(0, cx - roi_w // 2)
    y1 = max(0, cy - roi_h // 2)
    x2 = min(w, cx + roi_w // 2)
    y2 = min(h, cy + roi_h // 2)

    # Strategy 1: GrabCut with definite foreground seed at (cx, cy)
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
    
    # Mark small circle around click point as definite foreground
    seed_rad = max(3, int(min(w, h) * 0.02))
    cv2.circle(mask, (cx, cy), seed_rad, cv2.GC_FGD, -1)

    # Borders as background
    mask[0, :] = cv2.GC_BGD
    mask[h-1, :] = cv2.GC_BGD
    mask[:, 0] = cv2.GC_BGD
    mask[:, w-1] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        if num_labels > 1:
            target_label = labels[cy, cx]
            if target_label == 0:
                # Find closest non-zero label within 5 pixels
                for r in range(1, 10):
                    sub = labels[max(0, cy-r):min(h, cy+r+1), max(0, cx-r):min(w, cx+r+1)]
                    non_zeros = sub[sub > 0]
                    if len(non_zeros) > 0:
                        target_label = int(non_zeros[0])
                        break

            if target_label > 0:
                comp_mask = (labels == target_label).astype('uint8')
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) >= 15:
                        eps = 0.005 * cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, eps, True)
                        if len(approx) >= 3:
                            return [{"x": round(float(p[0][0]) / w, 5), "y": round(float(p[0][1]) / h, 5)} for p in approx]
    except Exception as e:
        sys.stderr.write(f"GrabCut point error: {e}\n")

    # Strategy 2: Adaptive FloodFill from point
    try:
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        lo_diff = 24
        up_diff = 24
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(img, flood_mask, (cx, cy), 0, (lo_diff, lo_diff, lo_diff), (up_diff, up_diff, up_diff), flags)
        seg_mask = flood_mask[1:h+1, 1:w+1]
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= 15:
                eps = 0.005 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True)
                if len(approx) >= 3:
                    return [{"x": round(float(p[0][0]) / w, 5), "y": round(float(p[0][1]) / h, 5)} for p in approx]
    except Exception as e:
        sys.stderr.write(f"Floodfill point error: {e}\n")

    # Strategy 3: Bounding box fallback
    pad_w = min(0.06, 30.0 / w)
    pad_h = min(0.06, 30.0 / h)
    return [
        {"x": round(max(0.0, px - pad_w), 5), "y": round(max(0.0, py - pad_h), 5)},
        {"x": round(min(1.0, px + pad_w), 5), "y": round(max(0.0, py - pad_h), 5)},
        {"x": round(min(1.0, px + pad_w), 5), "y": round(min(1.0, py + pad_h), 5)},
        {"x": round(max(0.0, px - pad_w), 5), "y": round(min(1.0, py + pad_h), 5)}
    ]

def segment_by_box(img, x1, y1, x2, y2):
    """
    Shrink-wrap a bounding box into a contour polygon.
    """
    h, w = img.shape[:2]
    px1 = max(0, min(w - 2, int(x1 * w)))
    py1 = max(0, min(h - 2, int(y1 * h)))
    px2 = max(px1 + 2, min(w - 1, int(x2 * w)))
    py2 = max(py1 + 2, min(h - 1, int(y2 * h)))

    bw = px2 - px1
    bh = py2 - py1
    if bw < 4 or bh < 4:
        return []

    mask = np.zeros((h, w), np.uint8)
    rect = (px1, py1, bw, bh)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

        roi = bin_mask[py1:py2, px1:px2]
        if np.sum(roi) < 10:
            return [
                {"x": round(x1, 5), "y": round(y1, 5)},
                {"x": round(x2, 5), "y": round(y1, 5)},
                {"x": round(x2, 5), "y": round(y2, 5)},
                {"x": round(x1, 5), "y": round(y2, 5)}
            ]

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            eps = 0.006 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) >= 3:
                return [{"x": round(float(p[0][0]) / w, 5), "y": round(float(p[0][1]) / h, 5)} for p in approx]
    except Exception as e:
        sys.stderr.write(f"GrabCut box error: {e}\n")

    return [
        {"x": round(x1, 5), "y": round(y1, 5)},
        {"x": round(x2, 5), "y": round(y1, 5)},
        {"x": round(x2, 5), "y": round(y2, 5)},
        {"x": round(x1, 5), "y": round(y2, 5)}
    ]

def main():
    parser = argparse.ArgumentParser(description="SAM & Intelligent Polygon Segmentation Bridge")
    parser.add_argument('--image', type=str, required=True, help="Input image path")
    parser.add_argument('--mode', type=str, choices=['point', 'box', 'full'], default='point', help="SAM mode")
    parser.add_argument('--point', type=str, help="Normalized point: 'x,y' (e.g. '0.5,0.5')")
    parser.add_argument('--box', type=str, help="Normalized box: 'x1,y1,x2,y2' (e.g. '0.2,0.2,0.8,0.8')")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"__SAM_RESULT__{json.dumps({'status': 'error', 'error': 'Image not found'})}", flush=True)
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"__SAM_RESULT__{json.dumps({'status': 'error', 'error': 'Cannot read image'})}", flush=True)
        return

    polygons = []

    if args.mode == 'point' and args.point:
        parts = [float(p.strip()) for p in args.point.split(',') if p.strip()]
        if len(parts) == 2:
            poly = segment_by_point(img, parts[0], parts[1])
            if poly and len(poly) >= 3:
                polygons.append(poly)
    elif args.mode == 'box' and args.box:
        parts = [float(p.strip()) for p in args.box.split(',') if p.strip()]
        if len(parts) == 4:
            poly = segment_by_box(img, parts[0], parts[1], parts[2], parts[3])
            if poly and len(poly) >= 3:
                polygons.append(poly)

    result_payload = {
        "status": "ok",
        "mode": args.mode,
        "count": len(polygons),
        "polygons": polygons
    }

    print(f"__SAM_RESULT__{json.dumps(result_payload)}", flush=True)

if __name__ == '__main__':
    main()
