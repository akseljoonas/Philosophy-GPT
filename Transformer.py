import math

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm

key = jax.random.PRNGKey(42)


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
        return self.vocab_size

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


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads combined together
    """

    head_num: int
    embed_dim: int

    def setup(self):
        self.heads = [
            SingleAttentionHead(
                embed_dim=self.embed_dim,
                head_size=self.embed_dim // self.head_num,
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
        self.layer1 = nn.Dense(features=(dim_mul * embed_dim), use_bias=False)
        self.layer2 = nn.Dense(features=embed_dim, use_bias=False)
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
        self.feedforward = FeedForward(
            embed_dim=self.embed_dim, dim_mul=self.dim_mul
        )

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
    block_layers: int

    def setup(self):
        self.token_embedding_table = nn.Embed(self.vocab_size, self.embed_dim)
        self.position_embedding_table = nn.Embed(
            self.context_length, self.embed_dim
        )
        #########################
        self.blocks = CustomSequential(
            [
                Block(
                    head_num=self.head_num,
                    embed_dim=self.embed_dim,
                    dim_mul=self.dim_mul,
                )
                for _ in range(self.block_layers)
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

    def generate(self, key, params, data, length):

        batch_size, _ = data.shape

        # Prepare jax.random.choice to operate on batches of data
        # points, without the need for explicit loops
        batched_random_choice = jax.vmap(jax.random.choice)

        for _ in range(length):

            # One new random key for every new character
            key, subkey = jax.random.split(key)

            # Prepare a (batch_size, 1) column of subkeys, one for every batch
            batched_key = subkey.reshape(1, -1)
            batched_key = jnp.repeat(batched_key, batch_size, axis=0)

            # Only use the last context_window characters to make predictions
            data_to_use = data[:, -self.context_length :]

            # Forward pass through the network to get the predictions
            logits = self.apply(
                {"params": params}, data_to_use, training=False
            )
            logits = logits[:, -1, :]
            probabilities = jax.nn.softmax(logits)

            # Preare a (batch_size, vocab_size) matrix storing token indexes
            token_indexes = jnp.arange(self.vocab_size).reshape(1, -1)
            token_indexes = jnp.repeat(token_indexes, batch_size, axis=0)

            # Selext new tokens for all batches based on probabilities
            next_indexes = batched_random_choice(
                batched_key, token_indexes, p=probabilities
            )
            next_indexes = next_indexes.reshape(batch_size, -1)

            # Append the new tokens to the sequence
            data = jnp.concatenate([data, next_indexes], axis=1)

        return data


# @jax.jit  # Jit the function for efficiency
def _train_step(state, batch, dropout_key):
    dropout_key, dropout_train_key = jax.random.split(dropout_key)

    def loss_fn(params):

        data, labels = batch

        # Same as model.apply
        logits = state.apply_fn(
            {"params": params},
            data,
            training=True,
            rngs={"dropout": dropout_train_key},
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
    # accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


# @jax.jit  # Jit the function for efficiency
def _eval_step(state, batch, training: bool):
    data, labels = batch
    logits = state.apply_fn({"params": state.params}, data, training)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mean_loss = jnp.mean(loss)
    # accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return mean_loss


def train(state, num_epochs, dropout_key):
    for epoch in tqdm(range(num_epochs)):
        train, train_labels = batch_loader.get_batch(
            key, batch_size, context_length, training=True
        )

        train_batch = (train, train_labels)
        state, train_loss = _train_step(state, train_batch, dropout_key)

        # We could use the loss and accuracy for logging here, e.g. in TensorBoard

        if epoch % 10 == 0:
            eval, eval_labels = batch_loader.get_batch(
                key, batch_size, context_length, training=False
            )
            eval_batch = (eval, eval_labels)
            eval_loss = _eval_step(state, eval_batch, training=False)

            print(
                f"Epoch {epoch}: Train loss {train_loss}, Eval loss {eval_loss}"
            )

    return state


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 4
    context_length = 8
    train_test_split_size = 0.9
    embed_dim = 32
    head_num = 2
    dim_mul = 2
    block_layers = 3
    learning_rate = 0.001
    max_iters = 500

    # Open the text file
    text = open_data()

    # Tokenize text
    tokenizer = Tokenizer(text=text, tokenizer_type="base")
    all_data = tokenizer.encode(text)

    # Initialize Batchloader
    batch_loader = BatchLoader(
        data=all_data, train_test_split_size=train_test_split_size
    )

    # Optimizer
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.01, peak_value=1, warmup_steps=100, decay_steps=2000
    )
    # optimizer = optax.adamw(scheduler)
    optimizer = optax.adamw(learning_rate=learning_rate)

    # Model init
    data = jnp.ones(
        (batch_size, context_length), dtype=jnp.int32
    )  # Example shape
    labels = jnp.ones((batch_size, context_length), dtype=jnp.int32)

    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=context_length,
        embed_dim=embed_dim,
        head_num=head_num,
        dim_mul=dim_mul,
        block_layers=block_layers,
    )

    # Specify what the key is used
    key, param_key, dropout_key = jax.random.split(key, num=3)
    variables = model.init(param_key, data=data, training=False)

    # Training
    params = variables["params"]

    class TrainState(train_state.TrainState):
        key: jax.Array

    state = TrainState.create(
        apply_fn=model.apply, params=params, key=dropout_key, tx=optimizer
    )

    trained_model_state = train(
        state=state, num_epochs=1000, dropout_key=dropout_key
    )

    # Generation
    key, subkey = jax.random.split(key, num=3)

    generated_seq = model.generate(
        key=subkey,
        params=trained_model_state.params,
        data=jax.numpy.ones((1, 1), dtype=jax.numpy.int32),
        length=500,
    )
    print(generated_seq)

    decoded_text = tokenizer.decode(generated_seq[0])

    print(decoded_text)
