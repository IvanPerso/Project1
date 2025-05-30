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

image_dir = Path("Fire-Smoke-Detection-Yolov11-2/test/images")
label_dir = Path("Fire-Smoke-Detection-Yolov11-2/test/labels")
image_paths = list(image_dir.glob("*.jpg"))

# ========================================
# 유틸 함수
# ========================================

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

# ========================================
# 평가 함수
# ========================================

def evaluate_model(model, image_paths, label_dir, iou_thresh=0.3): 
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    total_time = 0

    for img_path in tqdm(image_paths, desc="Evaluating"):
        img = Image.open(img_path)
        w, h = img.size
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_path, w, h)

        start = time.time()
        results = model(str(img_path), conf=0.1, iou=0.3, device=device) ##Iou,conf 낮춰 recall 높이기 나중에에
        end = time.time()
        total_time += (end - start)

        preds = results[0].boxes
        pred_by_class = defaultdict(list)
        for box in preds:
            cls = int(box.cls)
            if cls in [0, 1]:  # 0=Fire, 1=Smoke
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


recall, precision, avg_time_ms = evaluate_model(model, image_paths, label_dir)

print("\n📊 평가 결과")
print(f"  🔥 Fire   - Recall: {recall[0]:.4f}, Precision: {precision[0]:.4f}")
print(f"  💨 Smoke  - Recall: {recall[1]:.4f}, Precision: {precision[1]:.4f}")
print(f"  ⚡ 평균 추론 시간 : {avg_time_ms:.2f} ms")
