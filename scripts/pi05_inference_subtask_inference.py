import os
import numpy as np
import jax
import cv2
from flax import nnx
import time
from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
import jax.numpy as jnp
from openpi.training.config import get_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models.model import Observation
from openpi.models.pi0 import make_attn_mask

PALIGEMMA_EOS_TOKEN = 1
max_decoding_steps = 25
temperature = 0.1

### Step 1: Initialize model and load fine-tuned params
# Toggle between base model and fine-tuned model
USE_FINETUNED = True

print("Loading model...")
config = get_config("pi05_language_recovery")
model_rng = jax.random.key(0)
rng = jax.random.key(0)

if USE_FINETUNED:
    # Load from fine-tuned checkpoint (language recovery)
    checkpoint_path = config.checkpoint_dir / "499" / "params"
    print(f"Loading fine-tuned checkpoint from: {checkpoint_path}")
    params = _model.restore_params(checkpoint_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
else:
    # Load from base pretrained model
    model = config.model.create(model_rng)
    graphdef, state = nnx.split(model)
    loader = config.weight_loader
    params = nnx.state(model)
    params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    params_shape = params.to_pure_dict()
    loaded_params = loader.load(params_shape)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)

# JIT compile the decoding function once (first call will be slow)
model.jit_sample_low_level_task = nnx_utils.module_jit(model.sample_low_level_task, static_argnums=(3,))

# Tokenizer for encoding/decoding
tokenizer = PaligemmaTokenizer(max_len=50)

print("Model loaded successfully!")


def load_images(base_path, left_path, right_path):
    """Load 3 camera images and convert to model input format.

    Args:
        base_path: path to base camera image
        left_path: path to left wrist camera image
        right_path: path to right wrist camera image

    Returns:
        dict of image arrays in [-1, 1] range with batch dim
    """
    img_paths = {
        "base_0_rgb": base_path,
        "left_wrist_0_rgb": left_path,
        "right_wrist_0_rgb": right_path,
    }
    img_dict = {}
    for key, path in img_paths.items():
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not load {path}, using zeros")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img_dict[key] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0
    return img_dict


def build_observation(img_dict, high_prompt, low_prompt="PLACEHOLDER"):
    """Build an Observation from images and prompts.

    Args:
        img_dict: dict of image arrays from load_images()
        high_prompt: high-level task description
        low_prompt: placeholder low-level prompt (will be masked out for prediction)

    Returns:
        preprocessed Observation ready for inference
    """
    # Tokenize with high/low format to get proper masks
    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt(high_prompt, low_prompt)

    data = {
        'image': img_dict,
        'image_mask': {key: jnp.ones(1, dtype=jnp.bool_) for key in img_dict},
        'state': jnp.zeros((1, 32), dtype=jnp.float32),
        'tokenized_prompt': jnp.stack([tokens], axis=0),
        'tokenized_prompt_mask': jnp.stack([token_mask], axis=0),
        'token_ar_mask': jnp.stack([ar_mask], axis=0),
        'token_loss_mask': jnp.stack([loss_mask], axis=0),
    }
    observation = Observation.from_dict(data)
    rng_preprocess = jax.random.key(42)
    observation = _model.preprocess_observation(rng_preprocess, observation, train=False, image_keys=list(observation.images.keys()))

    # Mask out the low-level tokens (the model should predict these)
    lm = jnp.array(observation.token_loss_mask)
    new_tokenized_prompt = observation.tokenized_prompt.at[lm].set(0)
    new_tokenized_prompt_mask = observation.tokenized_prompt_mask.at[lm].set(False)
    observation = _model.Observation(
        images=observation.images,
        image_masks=observation.image_masks,
        state=observation.state,
        tokenized_prompt=new_tokenized_prompt,
        tokenized_prompt_mask=new_tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
    observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
    observation = jax.tree.map(jax.device_put, observation)
    return observation


def predict(observation):
    """Run subtask prediction on an observation.

    Returns:
        decoded text string of the predicted subtask
    """
    start_time = time.time()
    predicted_token, kv_cache, mask, ar_mask = model.jit_sample_low_level_task(
        rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
    )
    elapsed = time.time() - start_time

    # Decode predicted tokens to text
    pred_text = tokenizer.detokenize(np.array(predicted_token[0], dtype=np.int32))
    print(f"\033[31m[PRED]\033[0m {pred_text}")
    print(f"Time: {elapsed:.3f}s")
    return pred_text


### Step 2: Load default images and run initial inference
img_dir = './temp_doc/examples'
img_dict = load_images(
    os.path.join(img_dir, 'faceImg.png'),
    os.path.join(img_dir, 'leftImg.png'),
    os.path.join(img_dir, 'rightImg.png'),
)

# Run one warmup inference to trigger JIT compilation
print("\n--- Warmup inference (JIT compilation, will be slow) ---")
obs = build_observation(img_dict, "Pick up the flashcard on the table")
predict(obs)

# ============================================================
# Interactive pdb session
# ============================================================
# You can now try different inputs interactively:
#
#   # Try a different prompt with same images:
#   obs = build_observation(img_dict, "Open the drawer")
#   predict(obs)
#
#   # Load different images:
#   img_dict = load_images("path/to/base.png", "path/to/left.png", "path/to/right.png")
#   obs = build_observation(img_dict, "Stack the plates")
#   predict(obs)
#
#   # Try all 5 training prompts:
#   for p in ["Pick up the red cup", "Place the block on the shelf", "Open the drawer", "Wipe the table", "Stack the plates"]:
#       obs = build_observation(img_dict, p)
#       predict(obs)
#
#   # Continue execution:
#   c
# ============================================================
print("\n--- Dropping into interactive pdb. Use build_observation() and predict() to try inputs. ---")
import pdb; pdb.set_trace()
