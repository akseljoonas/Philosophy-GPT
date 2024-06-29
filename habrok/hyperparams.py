import jax

# Hyperparameters
batch_size = 4
context_length = 1024
train_test_split_size = 0.9
embed_dim = 768
n_heads = 12
mlp_dim_mul = 4  # between 2 and 8 according to UvA
n_blocks = 12
max_iters = 5000
learning_rate = 3e-4

# Scheduler
use_scheduler = False
scheduler_warmup_steps = 15  # typically 5-10% of the total training steps
scheduler_decay_steps = max_iters  #  Positive integer, the total length of the schedule
init_value = 3e-4
peak_value = 0.15

# Generation
temperature = 1
PROMPT = "The meaning of life is "
print(f"Prompt: {PROMPT}")

# Checkpoints
delete_checkpoints = False
CHECKPOINT_PATH = "/home1/s4790820/llm/Philosophy-GPT/habrok/checkpoints"
DATA_PATH = "/home1/s4790820/llm/Philosophy-GPT/new_nietzsche.txt"

# Parallelising
devices = jax.local_devices()
print("\n\n\n\n")
print(devices)

# Check if hyperparams make sense
assert embed_dim % n_heads == 0
assert scheduler_decay_steps <= max_iters
assert scheduler_warmup_steps <= scheduler_decay_steps
