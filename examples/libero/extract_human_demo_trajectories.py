"""Extract EEF trajectories and actions from LIBERO human demo data (LeRobot format).

Reads parquet files from .cache/data/libero-openpi and saves per-task JSON files
with the same structure as the rollout evaluation output.

NOTE: 3D object positions are NOT available in the LeRobot human demo data.
The LeRobot format only stores robot state (eef_pos, eef_orient, gripper) and actions.
Object positions were not recorded during human demonstrations.
"""

import json
import logging
import pathlib

import pandas as pd

# Path to the LeRobot human demo dataset
DATA_DIR = pathlib.Path(".cache/data/libero-openpi")
OUTPUT_DIR = pathlib.Path(".cache/output/libero_output_json")


def extract_human_demos() -> None:
    output_dir = OUTPUT_DIR / "human_demos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load episode metadata to get task descriptions
    episodes_file = DATA_DIR / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_file) as f:
        for line in f:
            episodes.append(json.loads(line))

    info = json.loads((DATA_DIR / "meta" / "info.json").read_text())
    data_path_template = info["data_path"]  # e.g. "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    chunks_size = info["chunks_size"]

    # Group episodes by task description
    task_to_episodes: dict[str, list[dict]] = {}
    for ep in episodes:
        # Each episode has a list of tasks; use the first one as the prompt
        task_desc = ep["tasks"][0]
        task_to_episodes.setdefault(task_desc, []).append(ep)

    logging.info(f"Found {len(episodes)} episodes across {len(task_to_episodes)} tasks")

    for task_desc, task_episodes in task_to_episodes.items():
        rollouts = []

        for ep in task_episodes:
            ep_idx = ep["episode_index"]
            # Compute chunk index and build parquet path
            chunk_idx = ep_idx // chunks_size
            parquet_path = DATA_DIR / data_path_template.format(
                episode_chunk=chunk_idx, episode_index=ep_idx
            )

            if not parquet_path.exists():
                logging.warning(f"Missing parquet file: {parquet_path}")
                continue

            df = pd.read_parquet(parquet_path)

            # Extract EEF trajectory from state column
            # State is 8D: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
            states = df["state"].tolist()
            eef_trajectory = [s[:3].tolist() for s in states]

            # Extract actions (7D)
            actions = [a.tolist() for a in df["actions"].tolist()]

            rollouts.append({
                "episode_idx": ep_idx,
                "success": True,  # human demos are assumed successful
                "num_steps": len(eef_trajectory),
                "eef_trajectory": eef_trajectory,
                "actions": actions,
                # Object positions not available in LeRobot human demo data
                "object_positions": [],
            })

        # Save per-task JSON with the same structure as eval rollout output
        json_filepath = output_dir / f"{task_desc}.json"
        with open(json_filepath, "w") as f:
            json.dump({
                "task_description": task_desc,
                "source": "human_demo",
                "num_rollouts": len(rollouts),
                "success_rate": 1.0,
                "object_names": [],  # not available in LeRobot format
                "rollouts": rollouts,
            }, f, indent=2)

        logging.info(f"Saved {len(rollouts)} demos for '{task_desc}' -> {json_filepath}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_human_demos()
