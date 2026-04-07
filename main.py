import os
import tempfile
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flasgger import Swagger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging
from ultralytics import YOLO

from classification import load_classifier_model, load_classifier_with_classes, transform


BASE_DIR = Path(__file__).resolve().parent
PRINTED_MODEL_DIR = BASE_DIR / "support_scripts" / "ocr_checkpoints" / "printed" / "best"
HANDWRITTEN_MODEL_DIR = BASE_DIR / "support_scripts" / "ocr_checkpoints" / "handwritten" / "best"
NEW_PRINTED_MODEL_DIR = BASE_DIR / "support_scripts" / "ocr_checkpoints" / "new-printed" / "best"
CLASSIFIER_WEIGHTS = BASE_DIR / "ocr_type_classifier3.pth"
METER_CLASSIFIER_WEIGHTS = BASE_DIR / "meter_classifier.pth"
YOLO_OLD_WEIGHTS = BASE_DIR / "runs" / "segment" / "train3" / "weights" / "best.pt"
YOLO_NEW_WEIGHTS = BASE_DIR / "runs" / "segment" / "train5" / "weights" / "best.pt"
MIN_OUTPUT_DIGITS = 5
MAX_OUTPUT_DIGITS = 10
NUM_BEAMS = 5

app = Flask(__name__)
app.config["SWAGGER"] = {
    "title": "SECON Meter OCR API",
    "uiversion": 3,
}
swagger = Swagger(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

logging.set_verbosity_error()


def crop_meter_region(image: Image.Image, image_path: str) -> tuple[Image.Image, str]:
    meter_type = classify_meter_type(image)
    yolo_model = model_yolo_old if meter_type == "old" else model_yolo_new
    yolo_results = yolo_model(image_path)
    boxes = yolo_results[0].boxes.xyxy
    if boxes is None or len(boxes) == 0:
        return image, meter_type

    x1, y1, x2, y2 = map(int, boxes[0])
    cropped = image.crop((x1, y1, x2, y2))
    return cropped, meter_type


def normalize_text(text: str) -> str:
    return "".join(text.split())


def resolve_model_path(model_path: str | Path) -> str | Path:
    candidate = Path(model_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    project_relative = (BASE_DIR / candidate).resolve()
    if project_relative.exists():
        return project_relative

    return str(model_path)


class TrOCREngine:
    def __init__(self, model_path: str | Path, device: torch.device, name: str):
        self.name = name
        self.device = device
        resolved_model_path = resolve_model_path(model_path)
        processor_kwargs = {"use_fast": True}
        model_kwargs = {}

        if isinstance(resolved_model_path, Path):
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True

        self.processor = TrOCRProcessor.from_pretrained(resolved_model_path, **processor_kwargs)
        self.model = VisionEncoderDecoderModel.from_pretrained(resolved_model_path, **model_kwargs).to(device)
        self.model.eval()
        self.allowed_token_ids = self._build_allowed_token_ids()
        self.eos_token_ids = self._build_eos_token_ids()

    def _build_allowed_token_ids(self) -> list[int]:
        tokenizer = self.processor.tokenizer
        allowed_token_ids = set()

        special_token_ids = {
            token_id
            for token_id in [
                tokenizer.pad_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
            ]
            if token_id is not None
        }
        allowed_token_ids.update(special_token_ids)

        for token_id in range(tokenizer.vocab_size):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            normalized = token_text.replace(" ", "")
            if normalized and all(character.isdigit() for character in normalized):
                allowed_token_ids.add(token_id)

        if tokenizer.eos_token_id is not None:
            allowed_token_ids.add(tokenizer.eos_token_id)

        return sorted(allowed_token_ids)

    def _build_eos_token_ids(self) -> set[int]:
        tokenizer = self.processor.tokenizer
        return {
            token_id
            for token_id in [
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
            ]
            if token_id is not None
        }

    def _current_digit_count(self, input_ids: torch.Tensor) -> int:
        token_ids = input_ids.tolist()
        decoded = self.processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        normalized = "".join(character for character in decoded if character.isdigit())
        return len(normalized)

    def _prefix_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        current_digit_count = self._current_digit_count(input_ids)
        if current_digit_count >= MIN_OUTPUT_DIGITS:
            return self.allowed_token_ids
        return [token_id for token_id in self.allowed_token_ids if token_id not in self.eos_token_ids]

    def _select_candidate(self, generated_ids: torch.Tensor, sequence_scores: torch.Tensor | None) -> str:
        candidates = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        normalized_candidates = [normalize_text(candidate) for candidate in candidates]

        if sequence_scores is None:
            return normalized_candidates[0]

        best_index = 0
        best_value = None
        for index, candidate in enumerate(normalized_candidates):
            score = float(sequence_scores[index].item())
            digit_only = candidate.isdigit()
            length = len(candidate)
            in_range = MIN_OUTPUT_DIGITS <= length <= MAX_OUTPUT_DIGITS
            score += 0.35 if digit_only else -1.0
            score += 0.25 if in_range else -0.12 * min(abs(length - MIN_OUTPUT_DIGITS), abs(length - MAX_OUTPUT_DIGITS))
            score += min(length, MAX_OUTPUT_DIGITS) * 0.02

            if best_value is None or score > best_value:
                best_value = score
                best_index = index

        return normalized_candidates[best_index]

    def predict(self, image: Image.Image) -> tuple[str, str]:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generation_output = self.model.generate(
                pixel_values,
                prefix_allowed_tokens_fn=self._prefix_allowed_tokens,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_BEAMS,
                early_stopping=True,
                length_penalty=0.2,
                no_repeat_ngram_size=2,
                max_new_tokens=MAX_OUTPUT_DIGITS,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prediction = self._select_candidate(generation_output.sequences, generation_output.sequences_scores)
        return prediction, self.name


class OCRTypeClassifier:
    def __init__(self, weights_path: str | Path, device: torch.device):
        resolved_path = resolve_model_path(weights_path)
        self.model, class_names = load_classifier_with_classes(weights_path=resolved_path, device=device)
        self.class_names = class_names or ["handwritten", "printed"]
        self.device = device

    def predict_label(self, image: Image.Image) -> str:
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
        return self.class_names[predicted_index]


class CombinedOCREngine:
    def __init__(
        self,
        classifier_weights: str | Path,
        printed_model_path: str | Path,
        handwritten_model_path: str | Path,
        device: torch.device,
    ):
        self.name = "combined_local"
        self.classifier = OCRTypeClassifier(classifier_weights, device=device)
        self.printed_engine = TrOCREngine(printed_model_path, device=device, name="printed_local")
        self.handwritten_engine = TrOCREngine(handwritten_model_path, device=device, name="handwritten_local")

    def predict(self, image: Image.Image) -> tuple[str, str]:
        predicted_label = self.classifier.predict_label(image)
        engine = self.handwritten_engine if predicted_label == "handwritten" else self.printed_engine
        prediction, _ = engine.predict(image)
        return prediction, predicted_label


combined_engine = CombinedOCREngine(
    classifier_weights=CLASSIFIER_WEIGHTS,
    printed_model_path=PRINTED_MODEL_DIR,
    handwritten_model_path=HANDWRITTEN_MODEL_DIR,
    device=device,
)
new_printed_engine = TrOCREngine(NEW_PRINTED_MODEL_DIR, device=device, name="new_printed_local")
meter_classifier_model = load_classifier_model(weights_path=METER_CLASSIFIER_WEIGHTS, device=device)
model_yolo_old = YOLO(str(YOLO_OLD_WEIGHTS))
model_yolo_new = YOLO(str(YOLO_NEW_WEIGHTS))


def classify_meter_type(image: Image.Image) -> str:
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = meter_classifier_model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return "old" if predicted == 1 else "new"


def recognize_image(image: Image.Image, image_path: str) -> dict[str, object]:
    cropped_image, meter_type = crop_meter_region(image, image_path)
    if meter_type == "old":
        prediction, _ = combined_engine.predict(cropped_image)
    else:
        prediction, _ = new_printed_engine.predict(cropped_image)

    return {
        "meter_type": meter_type,
        "prediction": prediction,
    }


@app.get("/")
def index():
    return jsonify({"message": "API запущен", "docs": "/apidocs/"})


@app.route("/batch-process", methods=["POST"])
def batch_process():
    """
    Пакетная обработка OCR-моделями
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: images
        type: file
        required: true
        description: Одно или несколько изображений для OCR
        allowMultiple: true
    responses:
      200:
        description: Результаты OCR по каждой модели
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

            with Image.open(temp_file_path) as image_file:
                image = image_file.convert("RGB")

            results.append(
                {
                    "image": file.filename,
                    **recognize_image(image, temp_file_path),
                }
            )

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
