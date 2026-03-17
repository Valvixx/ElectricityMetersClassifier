import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCRDataset(Dataset):
    def __init__(self, images_dir, labels_path, processor, max_target_length):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []

        with open(labels_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_name = row["image"].strip()
                text = row["text"].strip()
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
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    processor = TrOCRProcessor.from_pretrained(args.base_model, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = args.max_target_length
    model.config.num_beams = 1

    train_dataset = OCRDataset(
        images_dir=args.train_images,
        labels_path=args.train_labels,
        processor=processor,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = None
    if args.val_images and args.val_labels:
        val_dataset = OCRDataset(
            images_dir=args.val_images,
            labels_path=args.val_labels,
            processor=processor,
            max_target_length=args.max_target_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum_steps

        train_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.4f}")

        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{args.epochs} - val loss: {val_loss:.4f}")

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)

    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print(f"Training finished. Final model saved to: {final_dir}")
    if val_loader is not None:
        print(f"Best model saved to: {output_dir / 'best'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on a custom OCR dataset.")
    parser.add_argument("--train-images", required=True, help="Path to train images directory")
    parser.add_argument("--train-labels", required=True, help="Path to train labels.csv")
    parser.add_argument("--val-images", help="Path to validation images directory")
    parser.add_argument("--val-labels", help="Path to validation labels.csv")
    parser.add_argument("--output-dir", default="ocr_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--base-model", default="microsoft/trocr-base-printed", help="Base TrOCR model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-target-length", type=int, default=16, help="Maximum OCR text length")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
