from typing import Optional
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.proc = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.available = True
        except Exception:
            self.proc, self.model, self.available = None, None, False

    def describe(self, image_path: str, max_new_tokens: int = 50) -> str:
        if not self.available:
            return "Uncaptioned image"
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.proc(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.proc.decode(out[0], skip_special_tokens=True)
        except Exception:
            return "Uncaptioned image"
