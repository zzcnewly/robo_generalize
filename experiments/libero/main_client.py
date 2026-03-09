#!/usr/bin/env python3
"""Evaluate DreamZero on LIBERO-plus benchmarks.

Bridges LIBERO-plus simulation environments with the DreamZero inference server
(socket_test_optimized_AR.py) via the roboarena WebSocket protocol.

Key adaptations:
  - LIBERO has 1 agentview + 1 wrist camera; DreamZero expects 2 exterior + 1 wrist.
    We duplicate agentview into both exterior camera slots.
  - LIBERO images are 256x256; DreamZero expects 180x320. We resize with padding.
  - DreamZero returns (N, 8) actions (7 joint + 1 gripper). We take the first 6 dims
    as delta EEF (pos + rot) and the last dim as gripper for LIBERO's 7D action space.
    This assumes the model outputs LIBERO-compatible actions when fine-tuned on LIBERO data.
  - LIBERO doesn't expose joint positions, so we send zeros for joint_position and
    pack EEF pos + axisangle into cartesian_position.

Usage:
    # Start DreamZero server:
    torchrun --nproc_per_node=2 socket_test_optimized_AR.py --port 8000 --model-path <path>

    # Run evaluation:
    python experiments/libero/main_client.py --host localhost --port 8000 --task-suite-name libero_spatial
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import uuid

import cv2
import imageio
import numpy as np
import tqdm
import tyro

# Add LIBERO-plus to path
_LIBERO_PLUS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "external" / "LIBERO-plus"
sys.path.insert(0, str(_LIBERO_PLUS_ROOT))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import roboticstoolbox as rtb

import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy

# The websockets library by default sends a ping every 20 seconds and
# expects a pong response within 20 seconds. However, the sever may not
# send a pong response immediately if it is busy processing a request.
# Increase the ping interval and timeout so that the client can wait
# for a longer time before closing the connection.
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        # Build connect kwargs — older websockets versions don't support ping_interval/ping_timeout
        connect_kwargs: dict = {"compression": None, "max_size": None}
        import inspect
        _sig = inspect.signature(websockets.sync.client.connect)
        if "ping_interval" in _sig.parameters:
            connect_kwargs["ping_interval"] = PING_INTERVAL_SECS
            connect_kwargs["ping_timeout"] = PING_TIMEOUT_SECS

        try:
            conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")

        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Notify server that we're calling the infer endpoint (as opposed to the reset endpoint)
        obs["endpoint"] = "infer"

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, reset_info: Dict) -> None:
        # Notify server that we're calling the reset endpoint (as opposed to the infer endpoint)
        reset_info["endpoint"] = "reset"

        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Server connection
    host: str = "localhost"
    port: int = 8000

    # LIBERO environment
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    # Action chunking: execute this many actions before replanning
    replan_steps: int = 24

    # Output
    video_out_path: str = ".cache/output/libero/videos"
    seed: int = 7

    # Image resolution sent to DreamZero server
    target_height: int = 180
    target_width: int = 320


def resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image maintaining aspect ratio and center-pad to target size."""
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def quat2axisangle(quat):
    """Convert quaternion to axis-angle (from robosuite)."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def extract_state(obs: dict):
    """Extract DreamZero-compatible state vectors from LIBERO observation.

    Returns:
        joint_position: (7,) float32 — zeros (LIBERO doesn't expose joints)
        cartesian_position: (6,) float32 — EEF pos (3) + axisangle (3)
        gripper_position: (1,) float32 — first gripper qpos
    """
    eef_pos = obs["robot0_eef_pos"]  # (3,)
    eef_axisangle = quat2axisangle(obs["robot0_eef_quat"])  # (3,)
    gripper_qpos = obs["robot0_gripper_qpos"]  # (2,)

    joint_position = np.zeros(7, dtype=np.float32)
    cartesian_position = np.concatenate([eef_pos, eef_axisangle]).astype(np.float32)
    gripper_position = np.array([gripper_qpos[0]], dtype=np.float32)
    return joint_position, cartesian_position, gripper_position


def build_observation(
    agentview: np.ndarray,
    wrist: np.ndarray,
    joint_position: np.ndarray,
    cartesian_position: np.ndarray,
    gripper_position: np.ndarray,
    prompt: str,
    session_id: str,
    target_h: int,
    target_w: int,
) -> dict:
    """Build a roboarena observation dict from LIBERO env outputs."""
    agentview_resized = resize_with_pad(agentview, target_h, target_w)
    wrist_resized = resize_with_pad(wrist, target_h, target_w)

    return {
        # Duplicate agentview into both exterior camera slots
        "observation/exterior_image_0_left": agentview_resized,
        "observation/exterior_image_1_left": agentview_resized.copy(),
        "observation/wrist_image_left": wrist_resized,
        "observation/joint_position": joint_position,
        "observation/cartesian_position": cartesian_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
        "session_id": session_id,
    }


def server_action_to_libero(action: np.ndarray, robot) -> list:
    """Convert DreamZero 8D action to LIBERO 7D action.

    DreamZero outputs: (7 joint_position + 1 gripper) = 8D
    LIBERO expects:    (3 delta_pos + 3 delta_rot + 1 gripper) = 7D
    """
    # Calculate Forward Kinematics
    # .fkine returns an SE3 object (4x4 transformation matrix)
    # print("End Effector Pose (Transformation Matrix):")
    T = robot.fkine(action[:7])
    pos = T.t
    rpy = T.rpy(unit='rad', order='zyx')
    return np.concatenate([pos, rpy]).tolist() + [action[-1].item()]


def parse_server_actions(result) -> np.ndarray:
    """Parse the server response into an (N, 8) action array."""
    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, dict):
        joint_act = None
        grip_act = None
        for k, v in result.items():
            if "joint_position" in k:
                joint_act = np.asarray(v)
            elif "gripper" in k:
                grip_act = np.asarray(v)
        if joint_act is None:
            raise ValueError(f"No joint_position key in server response: {list(result.keys())}")
        if joint_act.ndim == 1:
            joint_act = joint_act.reshape(1, -1)
        if grip_act is not None:
            if grip_act.ndim == 1:
                grip_act = grip_act.reshape(-1, 1)
            return np.concatenate([joint_act, grip_act], axis=-1)
        return joint_act

    raise ValueError(f"Unexpected server response type: {type(result)}")


def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment and return (env, task_description)."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    logging.info(f"Args: {args}")

    # Initialize LIBERO-plus benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Options: {list(MAX_STEPS.keys())}")

    # Connect to DreamZero server
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    robot = rtb.models.DH.Panda()
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            logging.info(f"Task {task_id}, Episode {episode_idx}: {task_description}")

            env.reset()
            session_id = str(uuid.uuid4())
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get images (rotate 180 degrees to match LIBERO training preprocessing)
                    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(agentview.copy())

                    if not action_plan:
                        # Query DreamZero server for new action chunk
                        joint_pos, cart_pos, grip_pos = extract_state(obs)
                        server_obs = build_observation(
                            agentview=agentview,
                            wrist=wrist,
                            joint_position=joint_pos,
                            cartesian_position=cart_pos,
                            gripper_position=grip_pos,
                            prompt=task_description,
                            session_id=session_id,
                            target_h=args.target_height,
                            target_w=args.target_width,
                        )

                        result = client.infer(server_obs)
                        # import pdb; pdb.set_trace()
                        actions = parse_server_actions(result)

                        assert len(actions) >= args.replan_steps, (
                            f"Need {args.replan_steps} replan steps but server returned {len(actions)} actions"
                        )

                        for a in actions[: args.replan_steps]:
                            action_plan.append(server_action_to_libero(a, robot))

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception at step {t}: {e}", exc_info=True)
                    break

            task_episodes += 1
            total_episodes += 1

            # Reset server state for next episode
            client.reset({})

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:80]
            video_path = pathlib.Path(args.video_out_path) / f"task{task_id:04d}_ep{episode_idx}_{suffix}.mp4"
            if replay_images:
                imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)

            logging.info(
                f"{'SUCCESS' if done else 'FAILURE'} | "
                f"Episodes: {total_episodes}, Successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {task_successes / task_episodes:.2f}")

    logging.info(f"Final success rate: {total_successes / total_episodes:.2f} ({total_episodes} episodes)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(eval_libero)
