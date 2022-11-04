from typing import Sequence
from abc import abstractmethod
from functools import partial
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import ml_collections
import optax


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def build_mlp(n_out_dims, n_layers=2, layer_size=256):
    return MLP(([layer_size] * n_layers) + [n_out_dims])


def make_apply_model(loss_fn):
    @jax.jit
    def apply_model(state, batch):
        batch_loss_fn = lambda params: loss_fn(state, params, batch)
        grad_fn = jax.value_and_grad(batch_loss_fn)
        loss, grads = grad_fn(state.params)
        return grads, loss

    return apply_model


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


class JAXModel(object):
    def __init__(self, n_in_dims, n_out_dims):
        self.n_in_dims = n_in_dims
        self.n_out_dims = n_out_dims
        self.apply_model = make_apply_model(self.loss_fn)

    @abstractmethod
    def build_net_arch(self):
        raise NotImplementedError

    def train_epoch(self, state, dataset):
        epoch_loss = []
        for batch in dataset.train_dataloader:
            grads, loss = self.apply_model(state, batch)
            state = update_model(state, grads)
            epoch_loss.append(loss)
        train_loss = np.mean(epoch_loss)
        return train_loss, state

    def create_train_state(self, config):
        network = self.build_net_arch()
        seed = np.random.randint(0, np.iinfo(int).max)
        rng = jax.random.PRNGKey(seed)
        params = network.init(rng, jnp.ones([1, self.n_in_dims]))["params"]
        tx = optax.adam(config.learning_rate)
        return train_state.TrainState.create(
            apply_fn=network.apply, params=params, tx=tx
        )

    def compute_val_loss(self, state, dataset):
        val_losses = []
        for batch in dataset.val_dataloader:
            _, loss = self.apply_model(state, batch)
            val_losses.append(loss)
        return np.mean(val_losses)

    def train(self, dataset, config):
        state = self.create_train_state(config)
        self.state = state
        best_val_loss = self.compute_val_loss(state, dataset)

        train_losses = []
        val_losses = []
        for epoch in range(config.num_epochs):
            train_loss, state = self.train_epoch(state, dataset)
            val_loss = self.compute_val_loss(state, dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.state = state

            if config.verbose:
                print(
                    "epoch:% 3d, train_loss: %.4f, val_loss: %.4f"
                    % (epoch, train_loss, val_loss)
                )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return self.state, train_losses, val_losses


class PrefModel(JAXModel):
    def __init__(self, n_embedding_dims, net_arch_kwargs={}):
        super().__init__(n_in_dims=n_embedding_dims, n_out_dims=2)
        self.net_arch_kwargs = net_arch_kwargs

    @partial(jax.jit, static_argnames=["self"])
    def _predict(self, state, params, a_embeddings, b_embeddings):
        param_dict = {"params": params}
        a_logits = state.apply_fn(param_dict, a_embeddings)
        b_logits = state.apply_fn(param_dict, b_embeddings)
        logsumexps = jax.scipy.special.logsumexp(
            jnp.concatenate([a_logits, b_logits], axis=1), axis=1, keepdims=True
        )
        win_logits = a_logits - logsumexps
        lose_logits = b_logits - logsumexps
        return jnp.concatenate([win_logits, lose_logits], axis=1)

    @partial(jax.jit, static_argnames=["self"])
    def loss_fn(self, state, params, batch):
        a_embeddings, b_embeddings, prefs = batch
        logits = self._predict(state, params, a_embeddings, b_embeddings)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, prefs)
        return jnp.mean(loss)

    @partial(jax.jit, static_argnames=["self"])
    def predict(self, a_embeddings, b_embeddings):
        logits = self._predict(
            self.state, self.state.params, a_embeddings, b_embeddings
        )
        return jnp.exp(logits[:, 0])

    @partial(jax.jit, static_argnames=["self"])
    def score(self, embeddings):
        param_dict = {"params": self.state.params}
        scores = self.state.apply_fn(param_dict, embeddings)
        return scores[:, 0]

    def build_net_arch(self):
        return build_mlp(n_out_dims=1, **self.net_arch_kwargs)
