from ultralytics import YOLO
import supervision as sv
import cv2
import os
import random

# A test script to detect objects in an image

# Load the model
model = YOLO(f"../runs{os.path.sep}obb{os.path.sep}train5{os.path.sep}weights{os.path.sep}best.pt")

# Get a random file from the test images
random_file = random.choice(os.listdir(f"CV2-9{os.path.sep}test{os.path.sep}images"))
file_name = os.path.join(f"CV2-9{os.path.sep}test{os.path.sep}images", random_file)

# Run the model on the image
results = model(file_name)
detections = sv.Detections.from_ultralytics(results[0])

# Create an annotator to draw the detections on the image
oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=cv2.imread(file_name),
    detections=detections
)
print(detections)

# Display the detections on the image
sv.plot_image(image=annotated_frame, size=(16, 16))