import json
import requests
import os
from collections import OrderedDict
from io import StringIO

import numpy as np
import torch

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import models
import datasets
import rlhf
import embed
import interrogator
import utils


embedding_model = embed.CLIP()

net_arch_kwargs = {"n_layers": 2, "layer_size": 256}
pref_model_train_config = rlhf.get_default_config()


def prefs_from_ranking(n):
    prefs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < 0.5:
                prefs.append((i, j, 0))
            else:
                prefs.append((j, i, 1))
    return prefs


def display_prompt(score, prompt):
    st.metric(label="score", value=np.round(float(score), decimals=2))
    st.code(prompt, language=None)


def display_losses(train_losses, val_losses):
    epochs = np.arange(0, len(train_losses), 1)
    data = np.stack([epochs, train_losses, val_losses], axis=1)
    cols = ["epochs", "training loss", "validation loss"]
    df = pd.DataFrame(data, columns=cols)
    st.line_chart(df, x="epochs")


def optimize_prompt(data):
    st.subheader("learn preferences")
    with st.spinner("fitting prompt scoring model to ranking..."):
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
        _, train_losses, val_losses = optimizer.train()
        display_losses(train_losses, val_losses)
    st.subheader("optimize prompt")
    with st.spinner(
        "searching for prompt that maximizes predicted preference score..."
    ):
        candidate_scores = np.array(pref_model.score(embeddings))
        scored_candidates = list(zip(prompts, candidate_scores))
        init_prompt = max(scored_candidates, key=lambda x: x[1])[0]
        score_func = lambda x: torch.tensor(np.array(pref_model.score(x)))
        gator = interrogator.Gator(
            embedding_model=embedding_model,
            score_func=score_func,
            display=display_prompt,
        )
        best_prompt = gator.search(init_prompt)
    return best_prompt


default_import = """two women in cosplay costumes eating a doughnut, trending on cg society, rayonism, cyborg - girl with silver hair, devours a hamburger
two women in cosplay costumes eating a doughnut, trending on cg society, rayonism, cyborg - girl with silver hair
two women in cosplay costumes eating a doughnut, trending on cg society, rayonism
two women in cosplay costumes eating a doughnut, trending on cg society
two women in cosplay costumes eating a doughnut"""

with st.form(key="form") as form:
    raw = st.text_area(
        label="paste ranked list of prompts here (one prompt per line):",
        value=default_import,
    )
    submit = st.form_submit_button("optimize prompt")

if submit:
    prompts = raw.split("\n")
    if len(prompts) >= 3:
        prefs = prefs_from_ranking(len(prompts))
        data = {"prompts": prompts, "prefs": prefs}
        opt_prompt = optimize_prompt(data)
    else:
        st.markdown("please rank at least 3 prompts")
