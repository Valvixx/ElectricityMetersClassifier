import csv
import random
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, TrOCRProcessor, VisionEncoderDecoderModel


TRAIN_IMAGES_DIR = "/home/valvixx/Downloads/Datasets/NewTrain/"
TRAIN_LABELS_PATH = "/home/valvixx/Downloads/Datasets/NewTrain/labels.csv"
VAL_IMAGES_DIR = "/home/valvixx/Downloads/Datasets/NewVal/"
VAL_LABELS_PATH = "/home/valvixx/Downloads/Datasets/NewVal/labels.csv"
OUTPUT_DIR = "ocr_checkpoints"
BASE_MODEL = "microsoft/trocr-base-printed"
RUN_NAME = "new-printed"
EPOCHS = 14
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MAX_TARGET_LENGTH = 16
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 0
FORCE_CPU = False
USE_AMP = True
FREEZE_ENCODER = False
USE_GRADIENT_CHECKPOINTING = True
EARLY_STOPPING_PATIENCE = 4
MIN_OUTPUT_DIGITS = 5
MAX_OUTPUT_DIGITS = 10
NUM_BEAMS = 5
TRAIN_WITH_AUGMENTATIONS = True
RANDOM_SEED = 42


@dataclass
class TrainConfig:
    train_images: str
    train_labels: str
    val_images: str | None
    val_labels: str | None
    output_dir: str
    run_name: str
    base_model: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_target_length: int
    grad_accum_steps: int
    num_workers: int
    cpu: bool
    use_amp: bool
    freeze_encoder: bool
    use_gradient_checkpointing: bool
    early_stopping_patience: int


class OCRDataset(Dataset):
    def __init__(self, images_dir, labels_path, processor, max_target_length, augment=False):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.samples = []

        with open(labels_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {labels_path}")

            normalized_fieldnames = {field.strip().lower(): field for field in reader.fieldnames}
            image_field = normalized_fieldnames.get("image")
            text_field = normalized_fieldnames.get("text")

            if image_field is None or text_field is None:
                available_fields = ", ".join(reader.fieldnames)
                raise ValueError(
                    f"CSV {labels_path} must contain 'image' and 'text' columns. "
                    f"Found: {available_fields}"
                )

            for row in reader:
                image_name = row[image_field].strip()
                text = normalize_text(row[text_field].strip())
                if image_name and text:
                    self.samples.append((image_name, text))

        if not self.samples:
            raise ValueError(f"No samples found in {labels_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, text = self.samples[index]
        image_path = self.images_dir / image_name

        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = apply_random_augmentations(image, index=index)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        label_ids = labels.clone()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "label_ids": label_ids,
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    label_ids = torch.stack([item["label_ids"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels, "label_ids": label_ids}


def normalize_text(text: str) -> str:
    return "".join(text.split())


def apply_random_augmentations(image: Image.Image, index: int) -> Image.Image:
    rng = random.Random(f"{RANDOM_SEED}:{index}:{random.random()}")
    working = image.convert("L")

    if rng.random() < 0.9:
        working = working.rotate(
            rng.uniform(-4.5, 4.5),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )

    if rng.random() < 0.7:
        width, height = working.size
        crop_x = int(width * rng.uniform(0.0, 0.04))
        crop_y = int(height * rng.uniform(0.0, 0.08))
        working = ImageOps.crop(working, border=(crop_x, crop_y, crop_x, crop_y))
        working = ImageOps.expand(working, border=(crop_x, crop_y, crop_x, crop_y), fill=255)

    if rng.random() < 0.85:
        working = ImageEnhance.Contrast(working).enhance(rng.uniform(0.7, 1.45))

    if rng.random() < 0.85:
        working = ImageEnhance.Brightness(working).enhance(rng.uniform(0.8, 1.2))

    if rng.random() < 0.45:
        working = ImageEnhance.Sharpness(working).enhance(rng.uniform(0.45, 1.8))

    if rng.random() < 0.35:
        working = working.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))

    if rng.random() < 0.3:
        working = working.filter(ImageFilter.MedianFilter(size=3))

    if rng.random() < 0.45:
        working = add_salt_and_pepper_noise(working, rng)

    return ImageOps.autocontrast(working).convert("RGB")


def add_salt_and_pepper_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    noisy = image.copy()
    pixels = noisy.load()
    width, height = noisy.size
    total_points = max(1, int(width * height * rng.uniform(0.0008, 0.004)))

    for _ in range(total_points):
        pixels[rng.randrange(width), rng.randrange(height)] = 0 if rng.random() < 0.5 else 255

    return noisy


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


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / max(len(data_loader), 1)


def evaluate_generation(model, processor, data_loader, device) -> tuple[float, float]:
    model.eval()
    total_cer = 0.0
    exact_matches = 0
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            label_ids = batch["label_ids"]
            generated_ids = model.generate(
                pixel_values,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                length_penalty=0.2,
                no_repeat_ngram_size=2,
                max_new_tokens=MAX_OUTPUT_DIGITS,
            )

            predictions = [
                normalize_text(text)
                for text in processor.batch_decode(generated_ids, skip_special_tokens=True)
            ]
            references = [
                normalize_text(text)
                for text in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            ]

            for reference, prediction in zip(references, predictions):
                total_cer += cer(reference, prediction)
                exact_matches += int(reference == prediction)
                sample_count += 1

    if sample_count == 0:
        return 1.0, 0.0
    return total_cer / sample_count, exact_matches / sample_count


def slugify_model_name(model_name: str) -> str:
    return model_name.rsplit("/", maxsplit=1)[-1].replace("_", "-")


def build_run_dir(config: TrainConfig) -> Path:
    output_root = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.run_name.strip():
        run_name = config.run_name.strip()
    else:
        dataset_name = Path(config.train_images).name or "dataset"
        run_name = f"{slugify_model_name(config.base_model)}_{dataset_name}_{timestamp}"

    return output_root / run_name


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    print(f"Using device: {device}")

    processor = TrOCRProcessor.from_pretrained(config.base_model, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.max_length = config.max_target_length
    model.generation_config.max_new_tokens = MAX_OUTPUT_DIGITS
    model.generation_config.num_beams = NUM_BEAMS
    model.generation_config.length_penalty = 0.2
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 2

    if config.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    if config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_dataset = OCRDataset(
        images_dir=config.train_images,
        labels_path=config.train_labels,
        processor=processor,
        max_target_length=config.max_target_length,
        augment=TRAIN_WITH_AUGMENTATIONS,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = None
    if config.val_images and config.val_labels:
        val_dataset = OCRDataset(
            images_dir=config.val_images,
            labels_path=config.val_labels,
            processor=processor,
            max_target_length=config.max_target_length,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    optimizer = Adafactor(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    run_dir = build_run_dir(config)
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    best_val_loss = None
    best_val_cer = None
    epochs_without_improvement = 0
    use_amp = device.type == "cuda" and config.use_amp
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(amp_device, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / config.grad_accum_steps

            scaler.scale(loss).backward()

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * config.grad_accum_steps

        train_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{config.epochs} - train loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            val_cer, val_exact = evaluate_generation(model, processor, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{config.epochs} - val loss: {val_loss:.4f} "
                f"- val CER: {val_cer:.4f} - val exact: {val_exact:.2%}"
            )

            improved = False
            if best_val_cer is None or val_cer < best_val_cer:
                improved = True
            elif best_val_cer is not None and abs(val_cer - best_val_cer) < 1e-6:
                improved = best_val_loss is None or val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                best_val_cer = val_cer
                epochs_without_improvement = 0
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                print(f"Updated best model: {best_dir}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    if val_loader is None:
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        print(f"Training finished. Model saved to: {best_dir}")
    else:
        print(f"Training finished. Best model saved to: {best_dir}")


def get_config() -> TrainConfig:
    return TrainConfig(
        train_images=TRAIN_IMAGES_DIR,
        train_labels=TRAIN_LABELS_PATH,
        val_images=VAL_IMAGES_DIR,
        val_labels=VAL_LABELS_PATH,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        base_model=BASE_MODEL,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_target_length=MAX_TARGET_LENGTH,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        num_workers=NUM_WORKERS,
        cpu=FORCE_CPU,
        use_amp=USE_AMP,
        freeze_encoder=FREEZE_ENCODER,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )


if __name__ == "__main__":
    train(get_config())
