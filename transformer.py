import cv2
from PIL import Image
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import supervision as sv
import os
from threading import Thread
from queue import Queue
import sys

q_parent_to_child = Queue()
q_child_to_parent = Queue()
q_child_to_parent.maxsize = 60

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

def thread_function():
    while (True):
        if q_parent_to_child.empty():
            continue
        while not q_parent_to_child.empty():
            frame = q_parent_to_child.get()
            # with torch.no_grad():
            #     inputs = image_processor(images=frame, return_tensors="pt").to(DEVICE)
            #     outputs = model(**inputs)

            #     target_sizes = torch.tensor([frame.shape[:2]]).to(DEVICE)
            #     results = image_processor.post_process_object_detection(
            #         outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
            #     )[0]

            # detect = sv.Detections.from_transformers(transformers_results=results)

            # ml_labels = []
            # for id, confidence in zip(detect.class_id, detect.confidence):
            #     string = str(model.config.id2label[id]) + " " + str(confidence)
            #     ml_labels.append(string)

            # frame = box_annotator.annotate(scene=frame, detections=detect, labels=ml_labels)
            while(q_child_to_parent.full()):
                pass
            q_child_to_parent.put_nowait(frame)

thread = Thread(target=thread_function, daemon=True)
thread.start()
count = 0
fps = 24
delay = 1000 // fps

try:
    while True:
        if count % 5 != 0:
            continue
        print(count)
        if (q_child_to_parent.full()):
            while not q_child_to_parent.empty():
                cv2.imshow('Live Object Detection', q_child_to_parent.get())
        else:
            ret, frame = vid.read()
            count+=1
            if not ret:
                break
            q_parent_to_child.put(frame)

            
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
finally:
    thread.join()
    vid.release()
    cv2.destroyAllWindows()

