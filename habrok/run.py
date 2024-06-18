import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from hyperparams import (
    CHECKPOINT_PATH,
    batch_size,
    context_length,
    delete_checkpoints,
    embed_dim,
    init_value,
    learning_rate,
    max_iters,
    mlp_dim_mul,
    n_blocks,
    n_heads,
    peak_value,
    scheduler_decay_steps,
    scheduler_warmup_steps,
    temperature,
    train_test_split_size,
    use_scheduler,
)
from model import TransformerModel
from tqdm.auto import tqdm
from utils import (
    BatchLoader,
    Tokenizer,
    TrainState,
    _eval_step,
    _train_step,
    generate,
    open_data,
    plot_loss_curves,
)

key = jax.random.PRNGKey(42)

# Open text
text = open_data()
len(text)


# Tokenizer
tokenizer = Tokenizer(text=text, tokenizer_type="gpt-4o")
all_data = tokenizer.encode(text)
print(tokenizer.get_vocab_size())

# print(tokenizer.decode(all_data[:100]))


# Batch loader
batch_loader = BatchLoader(
    data=all_data, train_test_split_size=train_test_split_size, key=key
)
train, targets = batch_loader.get_batch(batch_size, context_length, training=True)

# Optimizer
if use_scheduler:
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=scheduler_warmup_steps,
        decay_steps=scheduler_decay_steps,
    )
    optimizer = optax.adamw(scheduler)  # scheduler
else:
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
dir_path = CHECKPOINT_PATH

if delete_checkpoints:
    path = ocp.test_utils.erase_and_create_empty(dir_path)

if not os.path.exists(dir_path):
    os.mkdir(dir_path)


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
    path = dir_path
else:
    state = TrainState.create(
        apply_fn=model.apply, params=params, key=dropout_key, tx=optimizer
    )


# Checkpoints
options = ocp.CheckpointManagerOptions(max_to_keep=3)
checkpoint_manager = ocp.CheckpointManager(path, orbax_checkpointer, options)


# Training


def train(state, num_epochs, dropout_key):
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    train_losses = []
    eval_losses = []
    best_eval_loss = float("inf")

    for epoch in tqdm(range(num_epochs + 1)):
        # Get data
        train, train_labels = batch_loader.get_batch(
            batch_size, context_length, training=True
        )

        # Reshaping to be compatiple with pmap
        dropout_keys = jax.random.split(dropout_key, jax.local_device_count())
        train = train.reshape((jax.local_device_count(), -1, *train.shape[1:]))
        train_labels = train_labels.reshape(
            (jax.local_device_count(), -1, *train_labels.shape[1:])
        )

        # Train step
        train_batch = (train, train_labels)
        replicated_state, gpu_train_losses = _train_step(
            replicated_state, train_batch, dropout_keys
        )

        # Get mean loss across gpus
        train_loss = jnp.mean(gpu_train_losses)

        if epoch % 100 == 0:
            # Get data
            eval, eval_labels = batch_loader.get_batch(
                batch_size, context_length, training=False
            )

            # Reshaping to be compatiple with pmap
            eval = eval.reshape((jax.local_device_count(), -1, *eval.shape[1:]))
            eval_labels = eval_labels.reshape(
                (jax.local_device_count(), -1, *eval_labels.shape[1:])
            )

            # Eval step
            eval_batch = (eval, eval_labels)
            gpu_eval_losses = _eval_step(replicated_state, eval_batch)

            # Get mean loss across gpus
            eval_loss = jnp.mean(gpu_eval_losses)

            # Saving best model according to loss
            if eval_loss < best_eval_loss:
                print(f"Saved model with loss {eval_loss}")
                ckpt = {"model": jax.device_get(replicated_state)}
                save_args = orbax_utils.save_args_from_target(ckpt)

                checkpoint_manager.save(
                    epoch,
                    ckpt,
                    save_kwargs={"save_args": save_args},
                )
                checkpoint_manager.wait_until_finished()
                best_eval_loss = eval_loss

            # Appending losses
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            print(f"Epoch {epoch}: Train loss {train_loss}, Eval loss {eval_loss}")

    return jax.device_get(replicated_state), train_losses, eval_losses


trained_model_state, train_losses, eval_losses = train(
    state=state, num_epochs=max_iters, dropout_key=dropout_key
)
plot_loss_curves(train_losses, eval_losses)


# Generation

PROMPT = jnp.ones(
    (jax.device_count(), 1, 1), dtype=jax.numpy.int32
)  # (device_count, 1, 1)

generated_seq = generate(
    trained_model_state,
    PROMPT,
    50,
    temperature,
)

decoded_text = tokenizer.decode(generated_seq[0])

print(decoded_text)
