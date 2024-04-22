from ultralytics import YOLO

model = YOLO('yolov8n-obb.yaml')
model.train(data='./dataset/data.yaml',epochs=1)
metrics = model.val()
path = model.export(format='onnx')
