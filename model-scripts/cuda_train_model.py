import torch
from ultralytics import YOLO
import os

# Check if CUDA is available on the system
if torch.cuda.is_available():
    # CUDA is available
    device = torch.device("cuda")
    print("CUDA is available.")
    print("Device name:", torch.cuda.get_device_name(0))  # Print the name of the GPU
    print("CUDA version:", torch.version.cuda)  # Print the CUDA version
    print("CUDNN version:", torch.backends.cudnn.version())  # Print the cuDNN version
else:
    # CUDA is not available
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")



# Create a model to train
model = YOLO("yolov8n-obb.yaml")
model.to(device)

# Train the model
results = model.train(data=f".{os.path.sep}CV2-9{os.path.sep}data.yaml", epochs=11, imgsz=640)
metrics = model.val()