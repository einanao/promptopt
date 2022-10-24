import jax
import jax.numpy as jnp
import ml_collections

from . import datasets


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
    ):
        if pref_model_train_config is None:
            pref_model_train_config = get_default_config()
        if embedding_dataset is None:
            embedding_dataset = datasets.embeddingDataset()
        if pref_dataset is None:
            pref_dataset = datasets.PrefDataset(self.embedding_dataset)

        self.pref_model = pref_model
        self.pref_model_train_config = pref_model_train_config
        self.embedding_dataset = embedding_dataset
        self.pref_dataset = pref_dataset

        if pref_dataset is not None:
            self.train()

    def train(self):
        self.pref_dataset.split()
        self.pref_model.train(self.pref_dataset, self.pref_model_train_config)

    def update(self, *args):
        for x in zip(*args):
            self.pref_dataset.append(*x)
        self.train()

    def append(self, *args):
        self.embedding_dataset.append(*args)

    def get_embeddings(self):
        return jnp.array(self.embedding_dataset.embeddings)

    def max(self, learning_rate=1e-3, n_iters=1000):
        derivative_fn = jax.jit(jax.grad(lambda x: self.pref_model.score(x)[0, 0]))
        embeddings = self.get_embeddings()
        scores = self.pref_model.score(embeddings)
        embedding = embeddings[jnp.argmax(scores), :][jnp.newaxis, :]
        for _ in range(n_iters):
            embedding += learning_rate * derivative_fn(embedding)
        return embedding[0, :]

    def predict(self, embedding):
        embeddings = self.get_embeddings()
        return self.pref_model.predict(
            jnp.repeat(embedding[jnp.newaxis, :], embeddings.shape[0], axis=0),
            embeddings,
        )
