from copy import deepcopy

import jax
import jax.numpy as jnp
import ml_collections

from . import datasets
from . import embed


def get_default_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 1e-2
    config.batch_size = 32
    config.num_epochs = 100
    config.verbose = False
    return config


class RLHF(object):
    def __init__(
        self,
        pref_model,
        pref_model_train_config=None,
        embedding_dataset=None,
        pref_dataset=None,
        embedding_model=None,
    ):
        if pref_model_train_config is None:
            pref_model_train_config = get_default_config()
        if embedding_dataset is None:
            embedding_dataset = datasets.EmbeddingDataset()
        if pref_dataset is None:
            pref_dataset = datasets.PrefDataset(self.embedding_dataset)
        if embedding_model is None:
            embedding_model = embed.CLIP()

        self.pref_model = pref_model
        self.pref_model_train_config = pref_model_train_config
        self.embedding_dataset = embedding_dataset
        self.pref_dataset = pref_dataset
        self.embedding_model = embedding_model

        if pref_dataset is not None:
            self.train()

    def train(self):
        self.pref_dataset.split()
        self.pref_model.train(self.pref_dataset, self.pref_model_train_config)

    def update(self, *args):
        for x in zip(*args):
            self.pref_dataset.append(*x)
        self.train()

    def append_prompt(self, prompt):
        embedding = self.embedding_model.embed(prompt)
        self.embedding_dataset.append(embedding)

    def get_embeddings(self):
        return jnp.array(self.embedding_dataset.embeddings)

    def max_embedding(self, learning_rate=1e-3, n_iters=1000, reg_const=1e2):
        embeddings = self.get_embeddings()
        embedding_var = jnp.var(embeddings, axis=0)
        scores = self.pref_model.score(embeddings)
        init_embedding = embeddings[jnp.argmax(scores), :][jnp.newaxis, :]
        embedding = deepcopy(init_embedding)

        def objective_fn(embedding):
            score = self.pref_model.score(embedding)[0, 0]
            dist_penalty = (((embedding - init_embedding) ** 2) / embedding_var).mean()
            return score - reg_const * dist_penalty

        derivative_fn = jax.jit(jax.grad(objective_fn))

        for _ in range(n_iters):
            embedding += learning_rate * derivative_fn(embedding)

        return embedding[0, :]

    def predict_prefs(self, embedding):
        embeddings = self.get_embeddings()
        return self.pref_model.predict(
            jnp.repeat(embedding[jnp.newaxis, :], embeddings.shape[0], axis=0),
            embeddings,
        )
