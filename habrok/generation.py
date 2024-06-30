import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from model import TransformerModel
from utils import (
    Tokenizer,
    TrainState,
    generate,
    open_data,
)

# Hyperparameters
batch_size = 4
context_length = 1024
train_test_split_size = 0.9
embed_dim = 768
n_heads = 12
mlp_dim_mul = 4  # between 2 and 8 according to UvA
n_blocks = 12
learning_rate = 3e-4
temperature = 1

PROMPT = "What is the meaning of life "
SAVED_CHECKPOINT_PATH = "/home1/s4790820/llm/Philosophy-GPT/habrok/saved_checkpoints"
key = jax.random.PRNGKey(42)

# Open text
text = open_data("/home1/s4790820/llm/Philosophy-GPT/new_nietzsche.txt")
print(f"text lenght {len(text)}")


# Tokenizer
tokenizer = Tokenizer(text=text, tokenizer_type="gpt-2")
print(tokenizer.get_vocab_size())


# Optimizer
optimizer = optax.adamw(learning_rate)

# Model init
data = jnp.ones((batch_size, context_length), dtype=jnp.int32)

model = TransformerModel(
    vocab_size=tokenizer.get_vocab_size(),
    context_length=context_length,
    embed_dim=embed_dim,
    head_num=n_heads,
    dim_mul=mlp_dim_mul,
    n_blocks=n_blocks,
)

# specify what the key is used
key, param_key, dropout_key = jax.random.split(key, num=3)
variables = model.init(param_key, data=data, training=False)

# Training State
params = variables["params"]
orbax_checkpointer = ocp.PyTreeCheckpointer()
dir_path = SAVED_CHECKPOINT_PATH

if len(os.listdir(dir_path)) > 0:  # If we have saved checkpoints
    subdirs = sorted((int(d) for d in os.listdir(dir_path)), reverse=True)
    best_model_dir = os.path.join(dir_path, str(subdirs[0]) + "/default/")
    print(f"Loaded state {best_model_dir}")
    empty_state = TrainState.create(
        apply_fn=model.apply,
        params=jax.tree_util.tree_map(np.zeros_like, params),
        key=dropout_key,
        tx=optimizer,
    )
    target = {"model": empty_state}
    state = orbax_checkpointer.restore(best_model_dir, item=target)["model"]
else:
    raise FileNotFoundError


# Generation
prompt_tokens = tokenizer.encode(PROMPT)
prompt = jnp.array(prompt_tokens).reshape((1, len(prompt_tokens)))
prompt = jnp.repeat(prompt, jax.device_count(), axis=0).reshape(
    (jax.device_count(), 1, len(prompt_tokens))
)

# generated_seq = generate(
#     state,
#     prompt,
#     100,
#     temperature,
# )

# decoded_text = tokenizer.decode(generated_seq[0])
# print("What is the meaning of life?\n")
# print(decoded_text)


# Stat
prompt_tokens = tokenizer.encode("")
prompt = jnp.array(prompt_tokens).reshape((1, len(prompt_tokens)))
prompt = jnp.repeat(prompt, jax.device_count(), axis=0).reshape(
    (jax.device_count(), 1, len(prompt_tokens))
)


generated_seq = generate(
    state,
    prompt,
    1000,
    temperature,
)
decoded_text = tokenizer.decode(generated_seq[0])
print("1000 tokens:\n")
print(decoded_text)
