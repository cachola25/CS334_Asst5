import torch
from ultralytics import YOLO

device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# Load a model
model = YOLO("yolov8n-obb.pt")
model.to(device)
results = model.train(data=f".\CV2-9\data.yaml", epochs=11, imgsz=640)
metrics = model.val()