import math

import jax.numpy as jnp
from flax import linen as nn


class SingleAttentionHead(nn.Module):
    embed_dim: int
    head_size: int

    def setup(self):
        self.key = nn.Dense(self.head_size, use_bias=False)
        self.query = nn.Dense(self.head_size, use_bias=False)
        self.value = nn.Dense(self.head_size, use_bias=False)
        self.dropout = nn.Dropout(rate=0.2)

    def __call__(self, data, training):
        k = self.key(data)  # from embed_dim to head_size (B,T,C)
        q = self.query(data)  # from embed_size to head_size (B,T,C)
        v = self.value(data)  # from embed_size to head_size (B,T,C)

        weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(
            self.head_size
        )  # (B,T,T)

        # Lower triangular mask matrix of the size B, T, C (same btw as attention)
        mask = jnp.tril(weights)

        # for every zero, make it to -inf
        weights = nn.softmax(
            jnp.where(mask == 0, -9e16, weights), axis=-1
        )  # axis=-1 since we only want to softmax for each row of T not for the whole data as a whole

        weights = self.dropout(weights, deterministic=not training)

        attention = jnp.matmul(weights, v)  # (B,T,C)

        return attention


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads combined together
    """

    head_num: int
    embed_dim: int

    def setup(self):
        self.heads = [
            SingleAttentionHead(
                embed_dim=self.embed_dim, head_size=self.embed_dim // self.head_num
            )
            for _ in range(self.head_num)
        ]
        self.think = nn.Dense(self.embed_dim, use_bias=False)
        self.dropout = nn.Dropout(rate=0.2)

    def __call__(self, data, training):
        multiple_attentions = jnp.concatenate(
            [head(data, training) for head in self.heads], axis=-1
        )
        thoughts = self.think(multiple_attentions)
        out = self.dropout(thoughts, deterministic=not training)
        return out


class FeedForward(nn.Module):
    """Simple Feed Forward NN that goes from embed_dim to a higher dimension and then back to embed_dim"""

    embed_dim: int
    dim_mul: int

    def setup(self):
        # this is the heavy thinking part of the model, where it tries to make sense of what was learned
        # in the attention cycle lol
        self.layer1 = nn.Dense(features=(self.dim_mul * self.embed_dim), use_bias=False)
        self.layer2 = nn.Dense(features=self.embed_dim, use_bias=False)
        self.dropout = nn.Dropout(rate=0.2)

    def __call__(self, data, training: bool):
        x = data
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        x = self.dropout(x, deterministic=not training)
        return x


class Block(nn.Module):
    """One run through a block, which consists of MultiheadAttention + Feedforward + Layer Normalisation"""

    dim_mul: int
    embed_dim: int
    head_num: int

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.multihead = MultiHeadAttention(
            head_num=self.head_num, embed_dim=self.embed_dim
        )
        self.feedforward = FeedForward(embed_dim=self.embed_dim, dim_mul=self.dim_mul)

    def __call__(self, data, training: bool):
        x = data
        x = x + self.multihead(self.norm1(x), training)
        x = x + self.feedforward(self.norm2(x), training)

        return x


class CustomSequential(nn.Module):
    layers: list

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class TransformerModel(nn.Module):
    vocab_size: int
    context_length: int
    embed_dim: int
    head_num: int
    dim_mul: int
    n_blocks: int

    def setup(self):
        self.token_embedding_table = nn.Embed(self.vocab_size, self.embed_dim)
        self.position_embedding_table = nn.Embed(self.context_length, self.embed_dim)
        #########################
        self.blocks = CustomSequential(
            [
                Block(
                    head_num=self.head_num,
                    embed_dim=self.embed_dim,
                    dim_mul=self.dim_mul,
                )
                for _ in range(self.n_blocks)
            ]
        )

        #########################
        self.norm = nn.LayerNorm()
        self.linear = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, data, training: bool = True):
        _, context_length = data.shape

        token = self.token_embedding_table(data)
        position = self.position_embedding_table(jnp.arange(context_length))

        embedded_data = token + position

        iteration_data = self.blocks(
            embedded_data, training
        )  # data after one iteration MH,FF (4,8,32)
        data_normalized = self.norm(iteration_data)
        final_data = self.linear(data_normalized)

        return final_data
