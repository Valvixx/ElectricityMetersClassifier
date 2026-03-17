from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from classification import load_classifier_model, transform


INPUT_DIR = Path("/home/valvixx/Downloads/Meters/")
OUTPUT_DIR = Path("/home/valvixx/Downloads/Datasets/")
OLD_DIR_NAME = "Old"
NEW_DIR_NAME = "New"
CROP_PADDING = 0.15

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model = load_classifier_model(device=device)
model_yolo_old = YOLO("../runs/segment/train3/weights/best.pt")
model_yolo_new = YOLO("../runs/segment/train5/weights/best.pt")


def classify_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier_model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return "Old" if predicted == 1 else "New"


def build_output_path(output_dir, source_path):
    candidate = output_dir / source_path.name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = output_dir / f"{source_path.stem}_{counter}{source_path.suffix.lower()}"
        if not candidate.exists():
            return candidate
        counter += 1


def crop_meter(image, image_path, meter_type):
    model = model_yolo_old if meter_type == "Old" else model_yolo_new
    results = model(str(image_path))
    boxes = results[0].boxes.xyxy

    if boxes is None or len(boxes) == 0:
        return None

    x1, y1, x2, y2 = map(int, boxes[0])
    pad_x = int((x2 - x1) * CROP_PADDING)
    pad_y = int((y2 - y1) * CROP_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))


def process_images():
    old_dir = OUTPUT_DIR / OLD_DIR_NAME
    new_dir = OUTPUT_DIR / NEW_DIR_NAME
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(INPUT_DIR.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except UnidentifiedImageError:
            print(f"Skipping unsupported image: {image_path}")
            continue

        meter_type = classify_image(image)
        cropped_image = crop_meter(image, image_path, meter_type)

        if cropped_image is None:
            print(f"No detection found: {image_path}")
            continue

        destination_dir = old_dir if meter_type == "Old" else new_dir
        output_path = build_output_path(destination_dir, image_path)
        cropped_image.save(output_path)
        print(f"Saved {meter_type}: {output_path}")


if __name__ == "__main__":
    process_images()
