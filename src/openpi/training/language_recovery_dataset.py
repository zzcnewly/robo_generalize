"""In-memory dataset with 5 hand-made examples for recovering pi05 language output.

Each example has a high-level task prompt and a low-level subtask prompt,
along with images loaded from temp_doc/examples/. Actions are random since
the goal is to train the subtask generation (language) head only.
"""

import logging
import pathlib

import cv2
import numpy as np

import openpi.models.model as _model

logger = logging.getLogger("openpi")

# 5 hand-made examples: (high_level_prompt, low_level_prompt)
LANGUAGE_EXAMPLES = [
    (
        "Pick up the red cup from the table",
        "move the gripper above the cup",
    ),
    (
        "Place the block on the shelf",
        "lift the block upward slowly",
    ),
    (
        "Open the drawer on the left side",
        "grasp the drawer handle firmly",
    ),
    (
        "Wipe the table with the cloth",
        "move the cloth to the right side",
    ),
    (
        "Stack the plates on the counter",
        "align the plate with the stack below",
    ),
]


class LanguageRecoveryDataset:
    """In-memory dataset with 5 hand-made examples for language recovery training.

    Returns dicts with images, state, actions, prompt, and low_prompt.
    The TokenizeHighLowPrompt transform will convert prompts to tokens with
    the required ar_mask and loss_mask for subtask generation training.
    """

    def __init__(self, model_config: _model.BaseModelConfig, image_dir: str = "./temp_doc/examples"):
        self._model_config = model_config
        self._image_dir = pathlib.Path(image_dir)
        self._examples = LANGUAGE_EXAMPLES

        # Load and preprocess images once (shared across all examples for simplicity)
        self._images = self._load_images()
        logger.info(
            f"LanguageRecoveryDataset: {len(self._examples)} examples, "
            f"action_dim={model_config.action_dim}, action_horizon={model_config.action_horizon}"
        )

    def _load_images(self) -> dict[str, np.ndarray]:
        """Load the 3 camera images from disk and resize to 224x224."""
        # Map model image keys to available image files
        image_mapping = {
            "base_0_rgb": "faceImg.png",
            "left_wrist_0_rgb": "leftImg.png",
            "right_wrist_0_rgb": "rightImg.png",
        }
        images = {}
        for key, filename in image_mapping.items():
            img_path = self._image_dir / filename
            if img_path.exists():
                # Load image as BGR, convert to RGB, resize to 224x224
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                # Convert to float32 in [-1, 1] range
                images[key] = img.astype(np.float32) / 127.5 - 1.0
            else:
                # Use zero image if file not found
                logger.warning(f"Image not found: {img_path}, using zeros")
                images[key] = np.zeros((224, 224, 3), dtype=np.float32)
        return images

    def __getitem__(self, index: int) -> dict:
        """Return a single training example as a dict."""
        # Cycle through the 5 examples
        idx = index % len(self._examples)
        high_prompt, low_prompt = self._examples[idx]

        # Random state and actions (actions don't matter for language recovery)
        rng = np.random.RandomState(index)
        state = rng.uniform(-0.5, 0.5, size=(self._model_config.action_dim,)).astype(np.float32)
        actions = rng.uniform(-0.5, 0.5, size=(self._model_config.action_horizon, self._model_config.action_dim)).astype(
            np.float32
        )

        return {
            "image": {k: v.copy() for k, v in self._images.items()},
            "image_mask": {k: np.ones((), dtype=np.bool_) for k in self._images},
            "state": state,
            "actions": actions,
            # These will be consumed by TokenizeHighLowPrompt transform
            "prompt": high_prompt,
            "low_prompt": low_prompt,
        }

    def __len__(self) -> int:
        # Return a larger number so the DataLoader can create enough batches
        # The examples cycle via modulo in __getitem__
        return max(len(self._examples) * 200, 1000)
