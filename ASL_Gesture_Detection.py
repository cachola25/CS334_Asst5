import cv2
import os
from ultralytics import YOLO
import math

def draw_rotated_box_on_frame(frame, x, y, w, h, angle, color=(0, 255, 0), thickness=2):
    """
    Draw a rotated rectangle on a frame.

    Parameters:
        frame (numpy.ndarray): The frame to draw the rectangle on.
        x, y (int): The center coordinates of the rectangle.
        w (int): The width of the rectangle.
        h (int): The height of the rectangle.
        angle (float): The rotation angle of the rectangle in degrees.
        color (tuple): The color of the rectangle (BGR format).
        thickness (int): The thickness of the rectangle's edges.
    """
    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Calculate the vertices of the rotated rectangle
    x1 = int(x - w / 2 * math.cos(angle_rad) + h / 2 * math.sin(angle_rad))
    y1 = int(y - w / 2 * math.sin(angle_rad) - h / 2 * math.cos(angle_rad))
    x2 = int(x + w / 2 * math.cos(angle_rad) + h / 2 * math.sin(angle_rad))
    y2 = int(y + w / 2 * math.sin(angle_rad) - h / 2 * math.cos(angle_rad))
    x3 = int(2 * x - x1)
    y3 = int(2 * y - y1)
    x4 = int(2 * x - x2)
    y4 = int(2 * y - y2)

    # Draw the rotated rectangle on the frame
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x3, y3), color, thickness)
    cv2.line(frame, (x3, y3), (x4, y4), color, thickness)
    cv2.line(frame, (x4, y4), (x1, y1), color, thickness)

class_names = [
    "-", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z"
]
HOME = os.getcwd()
MODEL_PATH = "./runs/obb/train20/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.6

model = YOLO(MODEL_PATH)
results = model.predict(source="0", conf=CONFIDENCE_THRESHOLD, show=True)

# WIDTH = 640
# HEIGHT = 640
# vid = cv2.VideoCapture(0)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# try:
#     while True:
#         ret, frame = vid.read()
#         if not ret:
#             break
#         # results = model(frame)[0]

        

#         # for detection in results:
#         #     detection = detection.obb
#         #     confidence = detection.conf[0].item()
#         #     print(detection.names)

#         #     coords = detection.xywhr[0]
#         #     x, y, w, h, angle = coords.tolist()
#         #     draw_rotated_box_on_frame(frame, x, y, w, h, angle)

#         #     # if 0.6 > CONFIDENCE_THRESHOLD:
#         #     #     class_id = int(detection[5])
#         #     #     class_name = results.names[class_id]
               
#         #     #     # Draw class label
#         #     #     label = f"{class_name}: {confidence:.2f}"
#         #     #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # cv2.imshow('Live Object Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     vid.release()
#     cv2.destroyAllWindows()