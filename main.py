import os
import tempfile

import torch
from flask import Flask, jsonify, request
from flasgger import Swagger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging
from ultralytics import YOLO

from classification import load_classifier_model, transform

app = Flask(__name__)
app.config["SWAGGER"] = {
    "title": "SECON Meter OCR API",
    "uiversion": 3,
}
swagger = Swagger(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

logging.set_verbosity_error()

model_yolo_old = YOLO("runs/segment/train3/weights/best.pt")
model_yolo_new = YOLO("runs/segment/train5/weights/best.pt")
model_trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
classifier_model = load_classifier_model(device=device)


def classify_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier_model(img_t)
        predicted = torch.argmax(output, 1).item()
    return "Старый" if predicted == 1 else "Новый"


def recognize_meter_value(image, image_path):
    image_class = classify_image(image)
    yolo_model = model_yolo_old if image_class == "Старый" else model_yolo_new

    yolo_results = yolo_model(image_path)
    boxes = yolo_results[0].boxes.xyxy
    if boxes is None or len(boxes) == 0:
        return ""

    x1, y1, x2, y2 = map(int, boxes[0])
    cropped = image.crop((x1, y1, x2, y2))
    pixel_values = processor(images=cropped, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model_trocr.generate(pixel_values)

    recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return recognized_text.replace(" ", "").replace("\n", " ")


@app.get("/")
def index():
    return jsonify({"message": "API запущен", "docs": "/apidocs/"})


@app.route("/batch-process", methods=["POST"])
def batch_process():
    """
    Пакетная обработка изображений счетчиков
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: images
        type: file
        required: true
        description: Одно или несколько изображений счетчиков
        allowMultiple: true
    responses:
      200:
        description: Список распознанных значений
        schema:
          type: array
          items:
            type: string
      400:
        description: Ошибка валидации запроса
        schema:
          type: object
          properties:
            error:
              type: string
    """
    if "images" not in request.files:
        return jsonify({"error": "No image files provided"}), 400

    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "No images selected"}), 400

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            temp_file_path = os.path.join(temp_dir, file.filename)
            file.save(temp_file_path)

            image = Image.open(temp_file_path).convert("RGB")
            results.append(recognize_meter_value(image, temp_file_path))

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
