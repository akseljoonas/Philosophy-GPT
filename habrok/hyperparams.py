import jax


# Hyperparameters
batch_size = 16
context_length = 256
train_test_split_size = 0.9
embed_dim = 32
n_heads = 8
mlp_dim_mul = 4  # between 2 and 8 according to UvA
n_blocks = 6
max_iters = 100
learning_rate = 3e-4

# Scheduler
use_scheduler = False
scheduler_warmup_steps = 15  # typically 5-10% of the total training steps
scheduler_decay_steps = max_iters  #  Positive integer, the total length of the schedule
init_value = 3e-4
peak_value = 0.15

# Generation
temperature = 1

# Checkpoints
delete_checkpoints = True
CHECKPOINT_PATH = "/Users/akseljoonas/Documents/Kool/NN/Final Project/checkpoints"

# Parallelising
devices = jax.local_devices()
print(devices)

# Check if hyperparams make sense
assert embed_dim % n_heads == 0
assert scheduler_decay_steps <= max_iters
assert scheduler_warmup_steps <= scheduler_decay_steps
