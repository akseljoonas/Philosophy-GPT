import math

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm

key = jax.random.PRNGKey(42)

# Hyperparameters
batch_size = 8
context_length = 50
train_test_split_size = 0.9
embed_dim = 32
head_num = 4
dim_mul = 10


def open_data(path: str = "new_nietzsche.txt"):
    txt = open(path, "r", encoding="utf-8").read()
    return txt


class Tokenizer:
    """
    Class that takes care of encoding and decoding the text
    """

    def __init__(self, text: str, tokenizer_type: str = "base") -> None:
        self.tokenizer_type = tokenizer_type
        self.vocab_size, self.all_characters = self.sort_characters(text)

    def get_vocab_size(self):
        return jnp.copy(self.vocab_size)

    def sort_characters(self, data):
        all_characters = sorted(list(set(data)))
        vocab_size = len(all_characters)

        return vocab_size, all_characters

    def encode(self, text):
        encoded_text = []
        if self.tokenizer_type == "base":
            for c in text:
                num = self.all_characters.index(c)
                encoded_text.append(num)
        return jnp.array(encoded_text)

    def decode(self, encoded_text):
        text = []
        if self.tokenizer_type == "base":
            for n in encoded_text:
                char = self.all_characters[n]
                text.append(char)
            text = "".join([str(item) for item in text])

        return text


class BatchLoader:
    """Class that handles the actions to perform on the tokenized data"""

    def __init__(self, data, train_test_split_size) -> None:
        self.training_data, self.validation_data = self.splitting_data(
            data, train_test_split_size
        )

    def splitting_data(self, data, split_size):
        n = int(split_size * len(data))
        training_data = data[:n]
        validation_data = data[n:]
        return training_data, validation_data

    def get_batch(
        self, key, batch_size, context_length, training: bool = True
    ):
        train_batches = []
        target_batches = []

        if training:
            b_data = self.training_data
        else:
            b_data = self.validation_data

        for _ in range(batch_size):
            key, subkey = jax.random.split(key)
            pos = jax.random.randint(
                key=subkey,
                shape=(),
                minval=0,
                maxval=(len(b_data) - context_length),
            )
            batch_data = b_data[pos : pos + context_length]
            train_batches.append(batch_data)
            batch_data = b_data[pos + 1 : pos + context_length + 1]
            target_batches.append(batch_data)

        train_batches = jnp.stack(train_batches)
        target_batches = jnp.stack(target_batches)

        return train_batches, target_batches


class SingleAttentionHead(nn.Module):
    embed_dim: int
    head_size: int

    def setup(self):
        self.key = nn.Dense(self.head_size, use_bias=False)
        self.query = nn.Dense(self.head_size, use_bias=False)
        self.value = nn.Dense(self.head_size, use_bias=False)
        self.dropout = nn.Dropout(rate=0.2)

    def __call__(self, data, training):

        k = self.key(data)  # from embed_size to head_size (B,T,C)
        q = self.query(data)
        v = self.value(data)

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


if __name__ == "__main__":

    # Open the text file
    text = open_data()

    # Tokenize text
    tokenizer = Tokenizer(text=text, tokenizer_type="base")
    all_data = tokenizer.encode(text)

    # Initialize Batchloader
    batch_loader = BatchLoader(
        data=all_data, train_test_split_size=train_test_split_size
    )
