import cv2
import os
from ultralytics import YOLO

HOME = os.getcwd()
MODEL_PATH = "./runs/detect/train12/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.0
IOU_THRESHOLD = 0.8

model = YOLO(MODEL_PATH)
# model = YOLO("yolov8n.pt")

vid = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        result = model(frame)[0]
        
        for res in result.boxes.data.tolist():
            x1 = int(res[0])
            y1 = int(res[1])
            x2 = int(res[2])
            y2 = int(res[3])
            score = int(res[4])
            class_id = int(res[5])

            if score >= CONFIDENCE_THRESHOLD:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,result.names[class_id].upper(),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)

        
        cv2.imshow('Live Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    vid.release()
    cv2.destroyAllWindows()