import os
import json

from flask import render_template
from flask import request
from flask import Flask

import numpy as np
import torch

from promptopt import models
from promptopt import datasets
from promptopt import rlhf
from promptopt import embed
from promptopt import interrogator
from promptopt import utils


embedding_model = embed.CLIP()

net_arch_kwargs = {"n_layers": 2, "layer_size": 256}
pref_model_train_config = rlhf.get_default_config()
pref_model_train_config.verbose = True


def prefs_from_ranking(n):
    prefs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < 0.5:
                prefs.append((i, j, 0))
            else:
                prefs.append((j, i, 1))
    return prefs


def optimize_prompt(data):
    prompts = data["prompts"]
    pref_data = data["prefs"]
    pref_model = models.PrefModel(
        embedding_model.n_embedding_dims, net_arch_kwargs=net_arch_kwargs
    )
    embeddings = embedding_model.embed_strings(prompts)
    embedding_dataset = datasets.EmbeddingDataset(embeddings=list(embeddings))
    pref_dataset = datasets.PrefDataset(embedding_dataset, pref_data=pref_data)
    optimizer = rlhf.RLHF(
        pref_model, pref_model_train_config, embedding_dataset, pref_dataset
    )
    candidate_scores = np.array(pref_model.score(embeddings))
    scored_candidates = list(zip(prompts, candidate_scores))
    init_prompt = max(scored_candidates, key=lambda x: x[1])[0]
    score_func = lambda x: torch.tensor(np.array(pref_model.score(x)))
    gator = interrogator.Gator(embedding_model=embedding_model, score_func=score_func)
    # best_prompt = gator.search(init_prompt)
    best_prompt = init_prompt  # DEBUG
    return best_prompt


app = Flask(__name__)


@app.route("/api/opt_from_prefs", methods=["POST"])
def opt_from_prefs():
    data = json.loads(request.values["data"])
    prompt = optimize_prompt(data)
    resp = {"optimized_prompt": prompt}
    return json.dumps(resp)


@app.route("/api/opt_from_ranking", methods=["POST"])
def opt_from_ranking():
    data = json.loads(request.values["data"])
    data["prefs"] = prefs_from_ranking(len(data["prompts"]))
    prompt = optimize_prompt(data)
    resp = {"optimized_prompt": prompt}
    return json.dumps(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
