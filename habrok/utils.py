from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tiktoken
from flax.training import train_state
from hyperparams import context_length
from tqdm.auto import tqdm


def generate(state, data, length, temperature):
    params = state.params
    key = jax.random.PRNGKey(42)
    for _ in tqdm(range(length), miniters=length / 10):
        key, subkey = jax.random.split(key)
        generate_keys = jax.random.split(subkey, jax.local_device_count())
        data = _generate_step(state, generate_keys, data, params, temperature)

    return jax.device_get(data[0])


def open_data(folder_path="/home1/s4790820/llm/Philosophy-GPT/new_nietzsche.txt"):
    full_txt = open(folder_path, "r", encoding="utf-8").read()
    return full_txt


def plot_loss_curves(train_losses, eval_losses, eval_interval=100):
    epochs = range(len(train_losses))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, eval_losses, label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Curves")
    plt.legend()
    plt.savefig("/home1/s4790820/llm/Philosophy-GPT/habrok/loss_curves.png")
    plt.show()


@partial(jax.pmap, axis_name="batch")
@jax.jit  # Jit the function for efficiency
def _train_step(state, batch, dropout_key):
    dropout_key, dropout_train_key = jax.random.split(dropout_key)

    def loss_fn(params):
        data, labels = batch

        # Same as model.apply
        logits = state.apply_fn(
            {"params": params}, data, training=True, rngs={"dropout": dropout_train_key}
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mean_loss = jnp.mean(loss)

        return mean_loss, logits

    # Gradient function
    grad_fn = jax.value_and_grad(
        loss_fn,  # Function to calculate the loss
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, logits), grads = grad_fn(state.params)

    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


class TrainState(train_state.TrainState):
    key: jax.Array


@partial(jax.pmap, axis_name="batch")
@jax.jit  # Jit the function for efficiency
def _eval_step(state, batch):
    data, labels = batch
    logits = state.apply_fn({"params": state.params}, data, training=False)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mean_loss = jnp.mean(loss)

    return mean_loss


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4))
@jax.jit
def _generate_step(state, key, data, params, temperature):
    data_to_use = data[:, -context_length:]

    logits = state.apply_fn({"params": params}, data_to_use, training=False)
    logits = logits[:, -1, :]

    next_token = jax.random.categorical(key, logits / temperature, shape=(1, 1))

    data = jnp.concatenate((data, next_token), axis=1)
    return data


class Tokenizer:
    """
    Class that takes care of encoding and decoding the text
    """

    def __init__(self, text: str, tokenizer_type: str = "base") -> None:
        self.tokenizer_type = tokenizer_type

        if self.tokenizer_type == "base":
            self.vocab_size, self.all_characters = self.sort_characters(text)
        elif self.tokenizer_type == "gpt-2":
            self.enc = tiktoken.encoding_for_model("gpt-2")
            self.vocab_size = self.enc.n_vocab

    def get_vocab_size(self):
        return int(jnp.copy(self.vocab_size))

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
        elif self.tokenizer_type == "gpt-2":
            encoded_text = self.enc.encode(text)
        return jnp.array(encoded_text)

    def decode(self, encoded_text):
        text = []
        if self.tokenizer_type == "base":
            for n in encoded_text:
                char = self.all_characters[n]
                text.append(char)
            text = "".join([str(item) for item in text])

        elif self.tokenizer_type == "gpt-2":
            text = self.enc.decode(encoded_text)

        return text


class BatchLoader:
    def __init__(self, data, train_test_split_size, key) -> None:
        self.training_data, self.validation_data = self.splitting_data(
            data, train_test_split_size
        )
        self.key = key

    def splitting_data(self, data, split_size):
        n = int(split_size * len(data))
        training_data = data[:n]
        validation_data = data[n:]
        return training_data, validation_data

    def get_batch(self, batch_size, context_length, training: bool = True):
        train_batches = []
        target_batches = []

        if training:
            b_data = self.training_data
        else:
            b_data = self.validation_data

        for _ in range(batch_size):
            self.key, subkey = jax.random.split(self.key)
            pos = jax.random.randint(
                key=subkey, shape=(), minval=0, maxval=(len(b_data) - context_length)
            )
            batch_data = b_data[pos : pos + context_length]
            train_batches.append(batch_data)
            batch_data = b_data[pos + 1 : pos + context_length + 1]
            target_batches.append(batch_data)

        train_batches = jnp.stack(train_batches)
        target_batches = jnp.stack(target_batches)

        return train_batches, target_batches
