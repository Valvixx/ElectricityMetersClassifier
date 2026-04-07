from pathlib import Path
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42


class RandomImageAugmentations:
    def __call__(self, image):
        rng = random.Random(f"{RANDOM_SEED}:{random.random()}")
        working = image.convert("RGB")

        if rng.random() < 0.9:
            working = working.rotate(
                rng.uniform(-4.5, 4.5),
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255),
            )

        if rng.random() < 0.7:
            width, height = working.size
            crop_x = int(width * rng.uniform(0.0, 0.04))
            crop_y = int(height * rng.uniform(0.0, 0.08))
            working = ImageOps.crop(working, border=(crop_x, crop_y, crop_x, crop_y))
            working = ImageOps.expand(
                working,
                border=(crop_x, crop_y, crop_x, crop_y),
                fill=(255, 255, 255),
            )

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

        return ImageOps.autocontrast(working)


def add_salt_and_pepper_noise(image, rng):
    noisy = image.copy()
    pixels = noisy.load()
    width, height = noisy.size
    total_points = max(1, int(width * height * rng.uniform(0.0008, 0.004)))

    for _ in range(total_points):
        pixels[rng.randrange(width), rng.randrange(height)] = (
            0,
            0,
            0,
        ) if rng.random() < 0.5 else (255, 255, 255)

    return noisy


base_transform = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

transform = transforms.Compose(base_transform)
train_transform = transforms.Compose([RandomImageAugmentations(), *base_transform])


def create_classifier_model(device=None):
    return create_image_classifier_model(num_classes=2, device=device)


def create_image_classifier_model(num_classes, device=None):
    weights = models.ResNet18_Weights.DEFAULT
    classifier = models.resnet18(weights=weights)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return classifier.to(device)


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_classifier_model(weights_path="meter_classifier.pth", device=None):
    classifier, _ = load_classifier_with_classes(weights_path=weights_path, device=device)
    return classifier


def load_classifier_with_classes(weights_path="meter_classifier.pth", device=None, num_classes=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(resolve_path(weights_path), map_location=device)
    class_names = None

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        class_names = checkpoint.get("class_names")
    else:
        state_dict = checkpoint

    if num_classes is None:
        num_classes = len(class_names) if class_names else 2

    classifier = create_image_classifier_model(num_classes=num_classes, device=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier, class_names


def train_classifier(
    train_dir="data/train",
    val_dir="data/val",
    weights_path="meter_classifier.pth",
    epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    use_augmentations=False,
):
    classifier, _, _, _ = train_image_classifier(
        train_dir=train_dir,
        val_dir=val_dir,
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentations=use_augmentations,
    )
    return classifier


def train_image_classifier(
    train_dir,
    val_dir,
    weights_path,
    epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    use_augmentations=False,
):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform if use_augmentations else transform,
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    class_names = train_dataset.classes

    if val_dataset.classes != class_names:
        raise ValueError(
            f"Train/val classes mismatch. Train: {class_names}, val: {val_dataset.classes}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    classifier = create_image_classifier_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    best_accuracy = -1.0
    best_state_dict = None

    for epoch in range(epochs):
        classifier.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0
        scheduler.step(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }
        print(
            f"Epoch {epoch + 1}/{epochs} done. "
            f"Val accuracy: {accuracy:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in classifier.state_dict().items()
        }
    classifier.load_state_dict(best_state_dict)

    torch.save(
        {
            "state_dict": best_state_dict,
            "class_names": class_names,
        },
        resolve_path(weights_path),
    )
    print(f"Saved best checkpoint with val accuracy: {best_accuracy:.2%}")
    return classifier, class_names, train_dataset.class_to_idx, best_accuracy


if __name__ == "__main__":
    train_classifier()
