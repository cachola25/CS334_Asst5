from ultralytics import YOLO

# Define the class names
class_names = [
    "-", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z"
]

# Set the path to the model we trained
MODEL_PATH = "./best.pt"

# Set confidence threshold to only show predictions with a confidence of 80% or higher
CONFIDENCE_THRESHOLD = 0.7

# Load the model and make predictions on the webcam feed
model = YOLO(MODEL_PATH)
results = model.predict(source="1", conf=CONFIDENCE_THRESHOLD, show=True)