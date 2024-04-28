import torch
from ultralytics import YOLO
import os

# Train the model on the GPU if available, otherwise use the CPU
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# Create a model to train
model = YOLO("yolov8n-obb.yaml")
model.to(device)

# Train the model on the imgages in the data.yaml file
results = model.train(data=f".{os.path.sep}CV2-9{os.path.sep}data.yaml", epochs=250, imgsz=640)
metrics = model.val()