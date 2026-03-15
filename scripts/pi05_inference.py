"""
Pi0.5 inference script that runs a forward pass with user-given prompt, images, and states.
Outputs both autoregressive language tokens (from the LLM backbone) and flow-matching actions.

Usage:
    uv run scripts/pi05_inference.py \
        --config pi05_aloha \
        --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
        --prompt "pick up the cup" \
        --image_paths base_0_rgb=path/to/img.png \
        --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.0
"""

import dataclasses
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
import tyro

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.shared import download as _download
from openpi.training import config as _config

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class Args:
    # Training config name, must be a pi05 config (e.g., "pi05_aloha", "pi05_droid").
    config: str = "pi05_aloha"
    # Checkpoint directory or GCS path.
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base"

    # Text prompt describing the desired task.
    prompt: str = "pick up the object"
    # Image paths as key=value pairs (e.g., base_0_rgb=path/to/img.png).
    # Keys must match the model's expected image keys: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
    # Missing keys will be filled with zero images.
    image_paths: list[str] = dataclasses.field(default_factory=list)
    # Robot state as a list of floats (will be zero-padded to model action_dim).
    state: list[float] = dataclasses.field(default_factory=list)

    # Number of flow-matching denoising steps for action sampling.
    num_flow_steps: int = 10
    # Number of autoregressive language tokens to decode from the LLM backbone.
    num_language_tokens: int = 64


def load_images(image_paths: list[str], resolution: tuple[int, int]) -> dict[str, np.ndarray]:
    """Parse image_paths (key=path pairs) and load/resize them as float32 arrays in [-1, 1]."""
    from PIL import Image

    images = {}
    for kv in image_paths:
        if "=" not in kv:
            raise ValueError(f"Expected key=path format, got: {kv}")
        key, path = kv.split("=", 1)
        # Load image, resize to target resolution, convert to float32 [-1, 1]
        img = Image.open(path).convert("RGB").resize((resolution[1], resolution[0]))
        img_arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        images[key] = img_arr
    return images


def decode_language_tokens(
    model: Pi0,
    prefix_out: jax.Array,
    tokenizer_path: pathlib.Path,
    num_tokens: int,
) -> tuple[list[int], str]:
    """Decode LLM prefix output embeddings into language tokens via greedy argmax.

    Takes the last prefix output embedding and autoregressively decodes `num_tokens`
    by projecting through the Gemma embedder's decode (transpose of embedding table).
    """
    # Access the embedding table from the Linen module wrapped via nnx_bridge.ToNNX.
    # The bridge flattens Linen submodule params as NNX attributes, so the embedder's
    # input_embedding param is at `model.PaliGemma.llm.embedder['input_embedding'].value`.
    embed_table = model.PaliGemma.llm.embedder["input_embedding"].value
    embed_dim = embed_table.shape[1]

    # Greedy decode: take last prefix token output, project to vocab, take argmax
    token_ids = []
    last_hidden = prefix_out[:, -1:, :]  # shape: (1, 1, emb_dim)
    for _ in range(num_tokens):
        # Project hidden state to vocab logits via embedding table transpose
        logits = jnp.dot(last_hidden, embed_table.T)  # (1, 1, vocab_size)
        next_token = int(jnp.argmax(logits[0, 0]))
        # EOS token (id=1 for PaliGemma) signals end of generation
        if next_token == 1:
            break
        token_ids.append(next_token)
        # Re-embed the predicted token for the next step (Gemma scales by sqrt(embed_dim))
        last_hidden = embed_table[next_token][None, None, :] * jnp.sqrt(embed_dim)
        last_hidden = last_hidden.astype(prefix_out.dtype)

    # Decode token IDs to text with the PaliGemma sentencepiece tokenizer
    with tokenizer_path.open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    text = sp.decode(token_ids)
    return token_ids, text


def run_prefix_forward(
    model: Pi0,
    observation: _model.Observation,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the prefix-only forward pass through the LLM backbone.

    Returns the prefix output embeddings (from which language tokens can be decoded),
    along with the KV cache, prefix mask, and prefix AR mask for subsequent action sampling.
    """
    # Preprocess observation (image resizing, mask filling)
    observation = _model.preprocess_observation(None, observation, train=False)

    # Embed prefix: images + tokenized prompt (with discretized state for pi0.5)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    # Forward pass through LLM with prefix only
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    return prefix_out, kv_cache, prefix_mask, prefix_ar_mask


def run_action_sampling(
    model: Pi0,
    observation: _model.Observation,
    rng: jax.Array,
    *,
    num_steps: int = 10,
) -> np.ndarray:
    """Sample actions using the full model (prefix + suffix flow matching)."""
    actions = model.sample_actions(rng, observation, num_steps=num_steps)
    return np.asarray(actions)


def main(args: Args) -> None:
    # Load training config and validate it's a pi0.5 config
    train_config = _config.get_config(args.config)
    model_config = train_config.model
    if not getattr(model_config, "pi05", False):
        raise ValueError(f"Config '{args.config}' is not a pi0.5 config (pi05=False). Use a pi05_* config.")

    # Download checkpoint and load model parameters
    checkpoint_dir = _download.maybe_download(args.checkpoint_dir)
    logger.info("Loading model parameters from %s ...", checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = model_config.load(params)

    # Download the PaliGemma sentencepiece tokenizer (for decoding language output)
    tokenizer_path = _download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})

    # Prepare images: load user-provided images, fill missing keys with zeros
    resolution = _model.IMAGE_RESOLUTION
    user_images = load_images(args.image_paths, resolution)
    images = {}
    image_masks = {}
    for key in _model.IMAGE_KEYS:
        if key in user_images:
            # User provided this image
            images[key] = jnp.array(user_images[key])[None]  # add batch dim
            image_masks[key] = jnp.array([True])
        else:
            # Fill with zeros and mark as masked out
            images[key] = jnp.zeros((1, *resolution, 3), dtype=jnp.float32)
            image_masks[key] = jnp.array([False])

    # Prepare state: zero-pad to model action_dim
    state = (
        np.array(args.state, dtype=np.float32) if args.state else np.zeros(model_config.action_dim, dtype=np.float32)
    )
    if len(state) < model_config.action_dim:
        state = np.pad(state, (0, model_config.action_dim - len(state)))
    state = jnp.array(state)[None]  # add batch dim

    # Tokenize prompt with discretized state (pi0.5 format)
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer(model_config.max_token_len)
    # For pi0.5, state is discretized and included in the prompt tokens
    tokens, token_mask = tokenizer.tokenize(args.prompt, np.asarray(state[0]))
    tokenized_prompt = jnp.array(tokens)[None]  # add batch dim
    tokenized_prompt_mask = jnp.array(token_mask)[None]

    # Build the Observation struct
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )

    # === Step 1: Run prefix forward pass to get language output ===
    logger.info("Running prefix forward pass for language decoding...")
    prefix_out, kv_cache, prefix_mask, prefix_ar_mask = run_prefix_forward(model, observation)

    # Decode language tokens from the LLM backbone output
    token_ids, decoded_text = decode_language_tokens(model, prefix_out, tokenizer_path, args.num_language_tokens)

    print("\n" + "=" * 60)
    print("LANGUAGE OUTPUT (from LLM backbone)")
    print("=" * 60)
    print(f"  Prompt: {args.prompt}")
    print(f"  Decoded token IDs: {token_ids}")
    print(f"  Decoded text: {decoded_text}")

    # === Step 2: Sample actions via flow matching ===
    logger.info("Sampling actions via flow matching (%d steps)...", args.num_flow_steps)
    rng = jax.random.key(0)
    actions = run_action_sampling(model, observation, rng, num_steps=args.num_flow_steps)

    print("\n" + "=" * 60)
    print("ACTION OUTPUT (from flow matching)")
    print("=" * 60)
    print(
        f"  Action shape: {actions.shape} (batch, horizon={model_config.action_horizon}, dim={model_config.action_dim})"
    )
    print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  First timestep actions: {actions[0, 0, :]}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
