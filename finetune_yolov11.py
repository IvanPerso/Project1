from ultralytics import YOLO

# 모델 로드 및 fine-tuning 시작
model = YOLO("yolo11s.pt")  # 또는 yolov8n.pt

model.train(
    data="Fire-Smoke-Detection-Yolov11-2/data.yaml",
    epochs=500,
    imgsz=640,
    batch=128,
    device="5",
    name="yolov11-fire-smoke_epoch500",
    project="runs-fire-smoke",
    workers=8
)
