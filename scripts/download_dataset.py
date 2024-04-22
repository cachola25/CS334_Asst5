from roboflow import Roboflow
import os 

model_dir = f"../model-scripts{os.path.sep}CV2-9"
rf = Roboflow(api_key="pC0yrwt8JMUBIA3czOnY")
project = rf.workspace("221565zcv").project("cv2-a4ryn")
version = project.version(9)
dataset = version.download("yolov8-obb",location=model_dir)

class_names = [
    "-", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z"
]
class_indicies = {name: index for index, name in enumerate(class_names)}

img_width, img_height = 640,640

## PATH TO ANNOTATION FILES
valid_folder_path = f'{model_dir}{os.path.sep}valid{os.path.sep}labels'
train_folder_path = f'{model_dir}{os.path.sep}train{os.path.sep}labels'
labels_folder_path = f'{model_dir}{os.path.sep}test{os.path.sep}labels'


## NORMALIZES COORDINATES
def normalize_coordinates(coord, max_value):
    normalized_coord = float(coord) / max_value
    return normalized_coord

foldPaths = [valid_folder_path, train_folder_path, labels_folder_path]
## PROCESS EACH FILE
for curr_folder_path in foldPaths:
    for filename in os.listdir(curr_folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(curr_folder_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) == 10:
                    label = parts[-2]
                    coords = parts[:8]

                    class_index = class_indicies.get(label, -1)
                    if class_index != -1:
                        normalized_coordinates = [normalize_coordinates(coords[i], img_width if i % 2 == 0 else img_height) for i in range(8)]

                        new_line = f"{class_index} " + " ".join(map(str, normalized_coordinates))
                        new_lines.append(new_line)

            with open(file_path, 'w') as file:
                file.write('\n'.join(new_lines))