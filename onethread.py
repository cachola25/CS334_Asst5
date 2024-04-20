import cv2
from PIL import Image
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import supervision as sv
import os
from ultralytics import YOLO

HOME = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./runs/detect/train10/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8

model = YOLO(MODEL_PATH)

box_annotator = sv.BoxAnnotator()
vid = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            break



        detect = model(source=frame,show=True,conf=0.6,save_txt=True)

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