from ultralytics import YOLO
import supervision as sv
import cv2
import os
import random

model = YOLO('runs/obb/train20/weights/best.pt')


random_file = random.choice(os.listdir(f"CV2-9/test/images"))
file_name = os.path.join(f"CV2-9/test/images", random_file)

results = model(file_name)

detections = sv.Detections.from_ultralytics(results[0])

oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=cv2.imread(file_name),
    detections=detections
)
print(detections)

sv.plot_image(image=annotated_frame, size=(16, 16))