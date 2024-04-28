from roboflow import Roboflow
import os 

# Script to download the dataset from Roboflow
model_dir = f"..{os.path.sep}model-scripts{os.path.sep}CV2-9"
rf = Roboflow(api_key="pC0yrwt8JMUBIA3czOnY")
project = rf.workspace("221565zcv").project("cv2-a4ryn")
version = project.version(9)
dataset = version.download("yolov8-obb",location=model_dir) 