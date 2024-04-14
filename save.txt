import cv2
from PIL import Image
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import supervision as sv
import os

HOME = os.getcwd()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "facebook/detr-resnet-50"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

box_annotator = sv.BoxAnnotator()

vid = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        with torch.no_grad():
            inputs = image_processor(images=frame, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)

            target_sizes = torch.tensor([frame.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
            )[0]

        detect = sv.Detections.from_transformers(transformers_results=results)

        ml_labels = []
        for id, confidence in zip(detect.class_id, detect.confidence):
            string = str(model.config.id2label[id]) + " " + str(confidence)
            ml_labels.append(string)

        frame = box_annotator.annotate(scene=frame, detections=detect, labels=ml_labels)

        cv2.imshow('Live Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    vid.release()
    cv2.destroyAllWindows()