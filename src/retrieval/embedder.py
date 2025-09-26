from typing import List
import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class ImageEmbedder:
    def __init__(self, clip_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
    def encode_paths(self, paths: List[str]) -> List[List[float]]:
        embs = []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                t = self.preprocess(img).unsqueeze(0).to(self.device)
                e = self.model.encode_image(t)
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy().flatten().tolist())
        return embs
