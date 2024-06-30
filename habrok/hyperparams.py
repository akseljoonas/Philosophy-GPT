import jax

# Hyperparameters
batch_size = 4
context_length = 1024
train_test_split_size = 0.9
embed_dim = 768
n_heads = 12
mlp_dim_mul = 4  # between 2 and 8 according to UvA
n_blocks = 12
max_iters = 10000
learning_rate = 3e-4


# Generation
temperature = 1
PROMPT = "The meaning of life is "
print(f"Prompt: {PROMPT}")

# Checkpoints
delete_checkpoints = True
CHECKPOINT_PATH = "/home1/s4790820/llm/Philosophy-GPT/habrok/checkpoints"
DATA_PATH = "/home1/s4790820/llm/Philosophy-GPT/new_nietzsche.txt"

# Parallelising
devices = jax.local_devices()
print("\n\n\n\n")
print(devices)

# Check if hyperparams make sense
assert embed_dim % n_heads == 0
