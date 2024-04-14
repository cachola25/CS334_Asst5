from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
import os
import torch
import cv2
import supervision as sv
from PIL import Image
import numpy as np
import time

import matplotlib.pyplot as plt


HOME = os.getcwd()

# Load the image using PIL (Python Imaging Library)

vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    cv2.imwrite("test.jpg", frame)
    img_path = os.path.join(HOME, "test.jpg")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "facebook/detr-resnet-50"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.8

    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
    model.to(DEVICE)

    with torch.no_grad():
        # LOAD IMAGE AND PREDICT
        image = cv2.imread(img_path)
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
        )[0]

    detect = sv.Detections.from_transformers(transformers_results=results)


    ml_labels = []
    for id,confidence in zip(detect.class_id, detect.confidence):
        string = str(model.config.id2label[id]) + " " + str(confidence)
        ml_labels.append(string)

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detect,labels=ml_labels)

    # Assuming frame is a PIL Image object
    plt.imshow(frame)
    plt.axis("off")  # Turn off axis
    plt.show()

    classifier = pipeline("object-detection")
    results = classifier(img_path)

    # print(results)
    time.sleep(1)
