import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification import load_classifier_with_classes, transform  # noqa: E402


VAL_DIR = Path("/home/valvixx/Downloads/Datasets/OldVal/")
NEW_VAL_DIR = Path("/home/valvixx/Downloads/Datasets/NewVal/")
OUTPUT_CSV = SCRIPT_DIR / "ocr_eval_results" / "combined_local3.csv"
PRINTED_MODEL_DIR = SCRIPT_DIR / "ocr_checkpoints" / "printed" / "best"
HANDWRITTEN_MODEL_DIR = SCRIPT_DIR / "ocr_checkpoints" / "handwritten" / "best"
NEW_PRINTED_MODEL_DIR = SCRIPT_DIR / "ocr_checkpoints" / "new-printed" / "best"
CLASSIFIER_WEIGHTS = PROJECT_ROOT / "ocr_type_classifier3.pth"
FORCE_CPU = False
MIN_OUTPUT_DIGITS = 5
MAX_OUTPUT_DIGITS = 10
NUM_BEAMS = 5


@dataclass
class Sample:
    image_name: str
    text: str
    expected_type: str


def normalize_text(text: str) -> str:
    return "".join(text.split())


def load_samples(labels_path: Path, expected_type: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(labels_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {labels_path}")

        normalized_fieldnames = {field.strip().lower(): field for field in reader.fieldnames}
        image_field = normalized_fieldnames.get("image")
        text_field = normalized_fieldnames.get("text")
        if image_field is None or text_field is None:
            raise ValueError(f"CSV must contain image/text columns: {labels_path}")

        for row in reader:
            image_name = row[image_field].strip()
            text = row[text_field].strip()
            if image_name and text:
                samples.append(
                    Sample(
                        image_name=image_name,
                        text=normalize_text(text),
                        expected_type=expected_type,
                    )
                )

    if not samples:
        raise ValueError(f"No labeled samples found in {labels_path}")

    return samples


def load_all_samples() -> list[Sample]:
    samples = load_samples(VAL_DIR / "labels_printed.csv", expected_type="printed")
    samples.extend(load_samples(VAL_DIR / "labels_handwritten.csv", expected_type="handwritten"))
    return samples


def load_new_printed_samples() -> list[Sample]:
    return load_samples(NEW_VAL_DIR / "labels.csv", expected_type="new-printed")


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein_distance(reference, prediction) / len(reference)


def resolve_model_path(model_path: str | Path) -> str | Path:
    candidate = Path(model_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    project_relative = (PROJECT_ROOT / candidate).resolve()
    if project_relative.exists():
        return project_relative

    script_relative = (SCRIPT_DIR / candidate).resolve()
    if script_relative.exists():
        return script_relative

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
            # Prefer candidates that look like a plausible meter reading.
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


def evaluate_engine(engine, images_dir: Path, samples: list[Sample]) -> tuple[dict[str, float], list[dict[str, str]]]:
    exact_matches = 0
    total_distance = 0
    total_cer = 0.0
    rows: list[dict[str, str]] = []

    for sample in samples:
        image_path = images_dir / sample.image_name
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        prediction, route = engine.predict(image)
        distance = levenshtein_distance(sample.text, prediction)
        sample_cer = cer(sample.text, prediction)
        exact_matches += int(prediction == sample.text)
        total_distance += distance
        total_cer += sample_cer

        rows.append(
            {
                "engine": engine.name,
                "expected_type": sample.expected_type,
                "route": route,
                "image": sample.image_name,
                "expected": sample.text,
                "predicted": prediction,
                "exact_match": str(int(prediction == sample.text)),
                "edit_distance": str(distance),
                "cer": f"{sample_cer:.6f}",
            }
        )

    sample_count = len(samples)
    summary = {
        "samples": sample_count,
        "exact_match_rate": exact_matches / sample_count,
        "avg_edit_distance": total_distance / sample_count,
        "avg_cer": total_cer / sample_count,
    }
    return summary, rows


def save_detailed_results(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "engine",
                "expected_type",
                "route",
                "image",
                "expected",
                "predicted",
                "exact_match",
                "edit_distance",
                "cer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Using device: {device}")

    combined_engine = CombinedOCREngine(
        classifier_weights=CLASSIFIER_WEIGHTS,
        printed_model_path=PRINTED_MODEL_DIR,
        handwritten_model_path=HANDWRITTEN_MODEL_DIR,
        device=device,
    )
    combined_samples = load_all_samples()
    combined_summary, combined_rows = evaluate_engine(
        combined_engine,
        images_dir=VAL_DIR,
        samples=combined_samples,
    )

    new_printed_engine = TrOCREngine(NEW_PRINTED_MODEL_DIR, device=device, name="new_printed_local")
    new_printed_samples = load_new_printed_samples()
    new_printed_summary, new_printed_rows = evaluate_engine(
        new_printed_engine,
        images_dir=NEW_VAL_DIR,
        samples=new_printed_samples,
    )

    rows = [*combined_rows, *new_printed_rows]
    save_detailed_results(OUTPUT_CSV, rows)

    print(
        f"{combined_engine.name}: "
        f"exact_match_rate={combined_summary['exact_match_rate']:.2%}, "
        f"avg_edit_distance={combined_summary['avg_edit_distance']:.3f}, "
        f"avg_cer={combined_summary['avg_cer']:.3f}"
    )
    print(
        f"{new_printed_engine.name}: "
        f"exact_match_rate={new_printed_summary['exact_match_rate']:.2%}, "
        f"avg_edit_distance={new_printed_summary['avg_edit_distance']:.3f}, "
        f"avg_cer={new_printed_summary['avg_cer']:.3f}"
    )
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
