import open_clip
import torch


class CLIP(object):
    def __init__(self):
        self.model = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        )[0]

    def embed_text(self, text):
        tokens = open_clip.tokenize([text])
        with torch.no_grad(), torch.cuda.amp.autocast():
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding[0, :]
