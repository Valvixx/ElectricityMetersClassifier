from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from transformers.utils import logging
logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)

image = Image.open("../Images/img.png").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Recognized:", generated_text)
