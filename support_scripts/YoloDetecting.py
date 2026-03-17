from PIL import Image
import torch
from ultralytics import YOLO
import cv2

model = YOLO("../yolov8n.pt")

image_path = "../Images/84150_1712848001205.jpg"
image = Image.open(image_path)

results = model(image_path)

crops = []
for box in results[0].boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    cropped = image.crop((x1, y1, x2, y2))
    crops.append(cropped)

for i, crop in enumerate(crops):
    crop.save(f"crop_{i}.jpg")
