import time
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# ========================================
# 설정
# ========================================
device = 'cuda:5'  # GPU
model = YOLO("runs-fire-smoke/yolov8-fire-smoke_epoch500/weights/best.pt").to(device)

image_dir = Path("Fire-Smoke-Detection-Yolov11-2/valid/images")
label_dir = Path("Fire-Smoke-Detection-Yolov11-2/valid/labels")
image_paths = list(image_dir.glob("*.jpg"))


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def load_ground_truth(label_path, width, height):
    gt = defaultdict(list)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            x1 = (x_center - w/2) * width
            y1 = (y_center - h/2) * height
            x2 = (x_center + w/2) * width
            y2 = (y_center + h/2) * height
            gt[cls].append([x1, y1, x2, y2])
    return gt

import itertools
import pandas as pd

# 튜닝할 conf, iou 값 리스트
conf_list = [0.02, 0.04, 0.06, 0.08, 0.1]
iou_list = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.3, 0.32]

# 결과 저장
results = []

# 기존 평가 함수 일부 수정 (conf, iou 파라미터화)
def evaluate_model_conf_iou(model, image_paths, label_dir, conf, iou_thresh):
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    total_time = 0

    for img_path in image_paths:
        img = Image.open(img_path)
        w, h = img.size
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_path, w, h)

        start = time.time()
        results = model(str(img_path), conf=conf, iou=iou_thresh, device=device)
        end = time.time()
        total_time += (end - start)

        preds = results[0].boxes
        pred_by_class = defaultdict(list)
        for box in preds:
            cls = int(box.cls)
            if cls in [0, 1]:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                pred_by_class[cls].append(xyxy)

        for cls in [0, 1]:
            gt_cls_boxes = gt_boxes.get(cls, [])
            pred_cls_boxes = pred_by_class.get(cls, [])
            matched = set()

            for gt_box in gt_cls_boxes:
                found_match = False
                for i, pred_box in enumerate(pred_cls_boxes):
                    if i in matched:
                        continue
                    if iou(gt_box, pred_box) >= iou_thresh:
                        TP[cls] += 1
                        matched.add(i)
                        found_match = True
                        break
                if not found_match:
                    FN[cls] += 1

            FP[cls] += len(pred_cls_boxes) - len(matched)

    recall = {cls: TP[cls] / (TP[cls] + FN[cls]) if (TP[cls] + FN[cls]) > 0 else 0.0 for cls in [0, 1]}
    precision = {cls: TP[cls] / (TP[cls] + FP[cls]) if (TP[cls] + FP[cls]) > 0 else 0.0 for cls in [0, 1]}
    avg_time_ms = (total_time / len(image_paths)) * 1000

    return recall, precision, avg_time_ms

# 튜닝 시작
for conf_val, iou_val in itertools.product(conf_list, iou_list):
    recall, precision, time_ms = evaluate_model_conf_iou(model, image_paths, label_dir, conf_val, iou_val)
    results.append({
        'conf': conf_val,
        'iou': iou_val,
        'recall_fire': recall[0],
        'recall_smoke': recall[1],
        'precision_fire': precision[0],
        'precision_smoke': precision[1],
        'inference_ms': time_ms
    })

# 결과 테이블
df = pd.DataFrame(results)

# 최고 recall_f 기준 정렬
df_sorted = df.sort_values(by='recall_fire', ascending=False)

# 출력
print(df_sorted.to_string(index=False))