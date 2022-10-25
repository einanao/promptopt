import open_clip
import torch


class CLIP(object):
    def __init__(self):
        self.model = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        )[0]
        self.n_embedding_dims = 512

    def embed_string(self, s):
        return self.embed_strings([s])[0]

    def embed_strings(self, strs):
        tokens = open_clip.tokenize(strs)
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = self.model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings.half().detach().numpy()
