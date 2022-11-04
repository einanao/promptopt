from transformers import CLIPProcessor, CLIPModel
import torch


class CLIP(object):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.n_embedding_dims = self.embed_string("").size

    def embed_string(self, s):
        return self.embed_strings([s])[0]

    def embed_strings(self, strs):
        inputs = self.processor(
            text=strs, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model.get_text_features(**inputs)
        return outputs.half().detach().numpy()
