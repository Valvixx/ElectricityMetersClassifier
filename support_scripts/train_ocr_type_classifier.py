from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification import train_image_classifier


TRAIN_DIR = "/home/valvixx/Downloads/Datasets/OcrType/train"
VAL_DIR = "/home/valvixx/Downloads/Datasets/OcrType/val"
WEIGHTS_PATH = PROJECT_ROOT / "ocr_type_classifier3.pth"
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TRAIN_WITH_AUGMENTATIONS = True


def main() -> None:
    classifier, class_names, _, accuracy = train_image_classifier(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        weights_path=WEIGHTS_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        use_augmentations=TRAIN_WITH_AUGMENTATIONS,
    )
    print(f"Saved classifier to: {WEIGHTS_PATH.resolve()}")
    print(f"Classes: {class_names}")
    print(f"Final val accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
