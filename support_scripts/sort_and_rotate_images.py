from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_DIR = Path("/home/valvixx/Downloads/г Белинский/")
OUTPUT_DIR = Path("/home/valvixx/Downloads/Meters")


def build_output_path(output_dir, source_path, counter):
    stem = source_path.stem
    suffix = source_path.suffix.lower() or ".jpg"
    return output_dir / f"{stem}_{counter}{suffix}"


def process_images(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 1

    for source_path in sorted(input_dir.rglob("*")):
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            with Image.open(source_path) as image:
                processed = image.copy()
        except UnidentifiedImageError:
            print(f"Skipping unsupported image: {source_path}")
            continue

        width, height = processed.size
        if width > height:
            processed = processed.transpose(Image.Transpose.ROTATE_270)

        output_path = build_output_path(output_dir, source_path, counter)
        processed.save(output_path)
        print(f"Saved: {output_path}")
        counter += 1

if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)
