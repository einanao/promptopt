"""
forked from https://huggingface.co/spaces/pharma/CLIP-Interrogator
"""

import hashlib
import math
import numpy as np
import os
import pickle
import torch

from tqdm import tqdm

import utils


chunk_size = 2048
flavor_intermediate_count = 2048
batch_size = 32


class LabelTable:
    def __init__(self, labels, desc, embedding_model, score_func):
        self.labels = labels
        self.embeds = []

        self.embedding_model = embedding_model
        self.score_func = score_func

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = os.path.join(utils.DATA_DIR, f"{desc}.pkl")
        if desc is not None and os.path.exists(cache_filepath):
            with open(cache_filepath, "rb") as f:
                data = pickle.load(f)
                if data["hash"] == hash:
                    self.labels = data["labels"]
                    self.embeds = data["embeds"]

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None):
                self.embeds.extend(list(embedding_model.embed_strings(chunk.tolist())))

            with open(cache_filepath, "wb") as f:
                pickle.dump(
                    {"labels": self.labels, "embeds": self.embeds, "hash": hash}, f
                )

    def _rank(self, text_embeds, top_count=1):
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).float()
        scores = self.score_func(text_embeds.detach().numpy())[None, :]
        _, top_labels = scores.cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, top_count=1):
        if len(self.labels) <= chunk_size:
            tops = self._rank(self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels) / chunk_size))
        keep_per_chunk = int(chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            stop = min(start + chunk_size, len(self.embeds))
            tops = self._rank(self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start + i] for i in tops])
            top_embeds.extend([self.embeds[start + i] for i in tops])

        tops = self._rank(top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def load_list(filename):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        items = [line.strip() for line in f.readlines()]
    return items


class Gator(object):
    def __init__(self, embedding_model, score_func, display):
        sites = [
            "Artstation",
            "behance",
            "cg society",
            "cgsociety",
            "deviantart",
            "dribble",
            "flickr",
            "instagram",
            "pexels",
            "pinterest",
            "pixabay",
            "pixiv",
            "polycount",
            "reddit",
            "shutterstock",
            "tumblr",
            "unsplash",
            "zbrush central",
        ]
        trending_list = [site for site in sites]
        trending_list.extend(["trending on " + site for site in sites])
        trending_list.extend(["featured on " + site for site in sites])
        trending_list.extend([site + " contest winner" for site in sites])

        raw_artists = load_list(os.path.join(utils.DATA_DIR, "artists.txt"))
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        self.artists = LabelTable(artists, "artists", embedding_model, score_func)
        self.flavors = LabelTable(
            load_list(os.path.join(utils.DATA_DIR, "flavors.txt")),
            "flavors",
            embedding_model,
            score_func,
        )
        self.mediums = LabelTable(
            load_list(os.path.join(utils.DATA_DIR, "mediums.txt")),
            "mediums",
            embedding_model,
            score_func,
        )
        self.movements = LabelTable(
            load_list(os.path.join(utils.DATA_DIR, "movements.txt")),
            "movements",
            embedding_model,
            score_func,
        )
        self.trendings = LabelTable(
            trending_list, "trendings", embedding_model, score_func
        )

        self.embedding_model = embedding_model
        self.score_func = score_func
        self.display = display

    def rank_top(self, text_array):
        text_features = self.embedding_model.embed_strings(text_array)
        scores = self.score_func(text_features)
        _, top_labels = scores.cpu().topk(1, dim=-1)
        return text_array[top_labels[0].numpy()]

    def score_prompt(self, prompt):
        return self.score_func(self.embedding_model.embed_strings([prompt]))[0]

    def search(self, init_prompt):
        flaves = self.flavors.rank(flavor_intermediate_count)
        best_medium = self.mediums.rank(1)[0]
        best_artist = self.artists.rank(1)[0]
        best_trending = self.trendings.rank(1)[0]
        best_movement = self.movements.rank(1)[0]

        best_prompt = init_prompt
        best_score = self.score_prompt(best_prompt)
        self.display(best_score, best_prompt)

        def check(addition):
            nonlocal best_prompt, best_score
            prompt = best_prompt + ", " + addition
            score = self.score_prompt(prompt)
            if score > best_score:
                best_score = score
                best_prompt = prompt
                self.display(best_score, best_prompt)
                return True
            return False

        def check_multi_batch(opts):
            nonlocal best_prompt, best_score
            prompts = []
            n = 2 ** len(opts)
            for i in range(n):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

                if len(prompts) >= batch_size or i == n - 1:
                    prompt = self.rank_top(prompts)
                    score = self.score_prompt(prompt)
                    if score > best_score:
                        best_score = score
                        best_prompt = prompt
                        self.display(best_score, best_prompt)
                    prompts = []

        check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        extended_flavors = set(flaves)
        for _ in range(25):
            try:
                best = self.rank_top([f"{best_prompt}, {f}" for f in extended_flavors])
                flave = best[len(best_prompt) + 2 :]
                if not check(flave):
                    break
                extended_flavors.remove(flave)
            except:
                # exceeded max prompt length
                break

        return best_prompt
