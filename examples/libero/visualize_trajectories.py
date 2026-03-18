"""Visualize EEF trajectories from rollout and human demo JSON files in a top-down (X-Y) view.

Usage:
    uv run python examples/libero/visualize_trajectories.py \
        --prompt "pick up the black bowl between the plate and the ramekin and place it on the plate"

    # Optionally specify custom paths or skip human demos
    uv run python examples/libero/visualize_trajectories.py \
        --prompt "pick up the black bowl ..." \
        --rollout-dir .cache/output/libero_output_json \
        --human-dir .cache/output/libero_output_json/human_demos \
        --output-dir .cache/output/libero_output_json/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def load_json(filepath: pathlib.Path) -> dict | None:
    """Load a trajectory JSON file if it exists."""
    if not filepath.exists():
        logging.warning(f"File not found: {filepath}")
        return None
    with open(filepath) as f:
        return json.load(f)


def _normalize_trajectory(traj_xy: np.ndarray, ref_start: np.ndarray, ref_end: np.ndarray,
                          align_end: bool = True) -> np.ndarray:
    """Normalize a single trajectory via similarity transform.

    Args:
        traj_xy: (N, 2) array of X-Y positions
        ref_start: target start point (2,)
        ref_end: target end point (2,)
        align_end: if True, apply full similarity transform (translate + rotate + scale) to
                   map start->ref_start and end->ref_end. If False, only translate so
                   start->ref_start, preserving the original direction and scale.
    """
    s = traj_xy[0]

    if not align_end:
        # Translation only: align start, keep natural direction and end
        return traj_xy - s + ref_start

    e = traj_xy[-1]
    d_orig = e - s
    orig_len = np.linalg.norm(d_orig)

    if orig_len < 1e-8:
        # Degenerate trajectory (start == end), just translate to ref_start
        return traj_xy - s + ref_start

    d_target = ref_end - ref_start
    target_len = np.linalg.norm(d_target)

    # Compute scale and rotation
    scale = target_len / orig_len
    theta = np.arctan2(d_target[1], d_target[0]) - np.arctan2(d_orig[1], d_orig[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Apply: translate to origin, rotate, scale, translate to ref_start
    centered = traj_xy - s
    return (rot @ centered.T).T * scale + ref_start


def _compute_mean_start_end(rollouts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean start and end positions across rollouts."""
    starts, ends = [], []
    for r in rollouts:
        traj = np.array(r["eef_trajectory"])
        if len(traj) == 0:
            continue
        starts.append(traj[0, :2])
        ends.append(traj[-1, :2])
    return np.mean(starts, axis=0), np.mean(ends, axis=0)


def plot_trajectories_on_ax(
    ax, rollouts: list[dict], color: str, alpha: float, label_prefix: str,
    ref_start: np.ndarray | None = None, ref_end: np.ndarray | None = None,
    align_end: bool = True,
):
    """Plot normalized EEF trajectories on a given axis in top-down (X-Y) view.

    Args:
        ref_start/ref_end: reference start/end for normalization (defaults to group mean)
        align_end: if True, apply full similarity transform aligning both start and end.
                   If False, only translate to align starts (for failed trajectories).
    """
    # Collect raw trajectories
    raw_trajs = []
    for r in rollouts:
        traj = np.array(r["eef_trajectory"])
        if len(traj) == 0:
            continue
        raw_trajs.append(traj[:, :2])

    if not raw_trajs:
        return

    # Compute reference frame from this group if not provided
    if ref_start is None or ref_end is None:
        mean_s, mean_e = _compute_mean_start_end(rollouts)
        if ref_start is None:
            ref_start = mean_s
        if ref_end is None:
            ref_end = mean_e

    # Normalize and plot each trajectory
    for i, traj_xy in enumerate(raw_trajs):
        normed = _normalize_trajectory(traj_xy, ref_start, ref_end, align_end=align_end)
        label = f"{label_prefix} (n={len(rollouts)})" if i == 0 else None
        ax.plot(normed[:, 0], normed[:, 1], color=color, alpha=alpha, linewidth=0.8, label=label)

    # Draw unified start marker (circle)
    ax.scatter(ref_start[0], ref_start[1], color=color, marker="o", s=80,
               edgecolors="black", linewidths=0.8, zorder=10)
    # Draw end marker only if ends are aligned
    if align_end:
        ax.scatter(ref_end[0], ref_end[1], color=color, marker="X", s=80,
                   edgecolors="black", linewidths=0.8, zorder=10)


def _extract_trajectory_features(traj_xy: np.ndarray) -> np.ndarray:
    """Extract compact kinematic and geometric features from a 2D trajectory.

    Feature vector (13-dim):
      [0]  path_length          — total arc length of the trajectory
      [1]  displacement          — straight-line distance from start to end
      [2]  straightness          — displacement / path_length (1 = perfectly straight)
      [3]  mean_speed            — average step-to-step speed
      [4]  std_speed             — standard deviation of speed (smoothness indicator)
      [5]  max_speed             — peak speed
      [6]  mean_acceleration     — average magnitude of acceleration (speed changes)
      [7]  std_acceleration      — variability of acceleration
      [8]  mean_curvature        — average unsigned curvature (how much the path bends)
      [9]  std_curvature         — variability of curvature
      [10] total_turning_angle   — sum of absolute heading changes (total rotation)
      [11] n_direction_reversals — number of times heading change switches sign (hesitation)
      [12] duration              — trajectory length in timesteps (proxy for time)

    Args:
        traj_xy: (N, 2) array of X-Y positions.

    Returns:
        Feature vector of shape (13,).
    """
    n_features = 13
    if len(traj_xy) < 2:
        return np.zeros(n_features)

    # Step displacements and speeds
    deltas = np.diff(traj_xy, axis=0)  # (N-1, 2)
    step_lengths = np.linalg.norm(deltas, axis=1)  # (N-1,)

    # Path length and displacement
    path_length = step_lengths.sum()
    displacement = np.linalg.norm(traj_xy[-1] - traj_xy[0])
    straightness = displacement / path_length if path_length > 1e-12 else 1.0

    # Speed statistics
    mean_speed = step_lengths.mean()
    std_speed = step_lengths.std()
    max_speed = step_lengths.max()

    # Acceleration: change in speed between consecutive steps
    if len(step_lengths) >= 2:
        accelerations = np.abs(np.diff(step_lengths))
        mean_accel = accelerations.mean()
        std_accel = accelerations.std()
    else:
        mean_accel = 0.0
        std_accel = 0.0

    # Heading angles and curvature from consecutive displacement vectors
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])  # (N-1,)
    if len(headings) >= 2:
        # Heading changes (wrapped to [-pi, pi])
        dheading = np.diff(headings)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi

        # Curvature = |heading change| / step length (at midpoints)
        mid_lengths = 0.5 * (step_lengths[:-1] + step_lengths[1:])
        # Avoid division by zero for stationary segments
        safe_lengths = np.where(mid_lengths > 1e-12, mid_lengths, 1.0)
        curvatures = np.abs(dheading) / safe_lengths

        mean_curvature = curvatures.mean()
        std_curvature = curvatures.std()
        total_turning = np.abs(dheading).sum()

        # Direction reversals: sign changes in heading difference
        signs = np.sign(dheading)
        nonzero_signs = signs[signs != 0]
        n_reversals = np.sum(np.abs(np.diff(nonzero_signs)) > 0) if len(nonzero_signs) >= 2 else 0
    else:
        mean_curvature = 0.0
        std_curvature = 0.0
        total_turning = 0.0
        n_reversals = 0

    # Duration as number of timesteps
    duration = float(len(traj_xy))

    return np.array([
        path_length, displacement, straightness,
        mean_speed, std_speed, max_speed,
        mean_accel, std_accel,
        mean_curvature, std_curvature, total_turning,
        float(n_reversals), duration,
    ])


# Human-readable names for each feature dimension (used in PCA loading annotation)
_FEATURE_NAMES = [
    "path_len", "displacement", "straightness",
    "mean_spd", "std_spd", "max_spd",
    "mean_accel", "std_accel",
    "mean_curv", "std_curv", "total_turn",
    "n_reversals", "duration",
]


def _extract_features_for_rollouts(rollouts: list[dict]) -> np.ndarray:
    """Extract kinematic feature vectors for a list of rollouts.

    Returns:
        Feature matrix of shape (n_rollouts, 13).
    """
    features = []
    for r in rollouts:
        traj = np.array(r["eef_trajectory"])
        features.append(_extract_trajectory_features(traj[:, :2] if len(traj) > 0 else traj))
    return np.array(features)


def plot_velocity_pca(ax, human_rollouts: list[dict] | None, robot_rollouts: list[dict] | None,
                      method: str = "pca"):
    """Run PCA or t-SNE on kinematic features and scatter-plot human vs robot trajectories.

    Each trajectory is summarized by 13 kinematic/geometric features (speed, curvature,
    straightness, etc.). The chosen method reduces these to 2D for visualization.

    Args:
        method: "pca" for PCA (linear, shows explained variance) or
                "tsne" for t-SNE (nonlinear, better at revealing clusters).
    """
    # Gather all rollout groups and their labels/colors
    groups: list[tuple[str, str, list[dict]]] = []
    if human_rollouts:
        groups.append(("Human demo", "steelblue", human_rollouts))
    if robot_rollouts:
        successes = [r for r in robot_rollouts if r["success"]]
        failures = [r for r in robot_rollouts if not r["success"]]
        if successes:
            groups.append(("Policy success", "seagreen", successes))
        if failures:
            groups.append(("Policy failure", "tomato", failures))

    if not groups:
        return

    # Extract features for each group and stack into a single matrix
    all_features = []
    group_indices = []  # (label, color, start_idx, end_idx)
    for label, color, rollouts in groups:
        feats = _extract_features_for_rollouts(rollouts)
        start = len(all_features)
        all_features.extend(feats)
        group_indices.append((label, color, start, start + len(feats)))

    feature_matrix = np.array(all_features)

    # Z-score normalize each feature so PCA isn't dominated by scale differences
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds[stds < 1e-12] = 1.0  # avoid division by zero for constant features
    feature_matrix = (feature_matrix - means) / stds

    # Reduce to 2D using the chosen method
    n_samples = feature_matrix.shape[0]
    if method == "tsne" and n_samples >= 4:
        # Perplexity must be less than n_samples; use a reasonable default
        # Lazy import to avoid slow sklearn.manifold load when using PCA
        from sklearn.manifold import TSNE
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500, learning_rate="auto",
                    init="pca", random_state=42)
        projected = tsne.fit_transform(feature_matrix)
    else:
        # Fall back to PCA (also used when too few samples for t-SNE)
        n_components = min(2, n_samples, feature_matrix.shape[1])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(feature_matrix)

    # Scatter each group with distinct color and marker
    for label, color, start, end in group_indices:
        pts = projected[start:end]
        y_vals = pts[:, 1] if projected.shape[1] == 2 else np.zeros(len(pts))
        # Circles for human, red crosses for policy success, red triangles for policy failure
        if label == "Policy failure":
            marker = "v"
        elif label.startswith("Policy"):
            marker = "x"
        else:
            marker = "o"
        scatter_color = "red" if label.startswith("Policy") else color
        ax.scatter(pts[:, 0], y_vals,
                   color=scatter_color, alpha=0.6, s=40, marker=marker,
                   edgecolors="black" if marker == "o" else scatter_color, linewidths=0.5,
                   label=f"{label} (n={end - start})")

    # Set axis labels based on method
    if method == "tsne" and n_samples >= 4:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Trajectory Kinematics t-SNE")
    else:
        var_explained = pca.explained_variance_ratio_
        loadings = pca.components_  # (n_components, n_features)
        for i in range(min(2, len(var_explained))):
            # Find the top 2 features by absolute loading weight for this PC
            top_idx = np.argsort(np.abs(loadings[i]))[::-1][:2]
            top_names = ", ".join(_FEATURE_NAMES[j] for j in top_idx)
            label_text = f"PC{i+1} ({var_explained[i]:.1%} var; {top_names})"
            if i == 0:
                ax.set_xlabel(label_text)
            elif i == 1:
                ax.set_ylabel(label_text)
        ax.set_title("Trajectory Kinematics PCA")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def visualize(prompt: str, rollout_dir: str, human_dir: str, output_dir: str, embed_method: str = "pca"):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_filename = f"{prompt}.json"

    # Load rollout data (policy evaluation) and human demo data
    rollout_data = load_json(pathlib.Path(rollout_dir) / json_filename)
    human_data = load_json(pathlib.Path(human_dir) / json_filename)

    if rollout_data is None and human_data is None:
        logging.error("No data found for this prompt. Check the prompt text and directories.")
        return

    # Determine layout
    has_rollout = rollout_data is not None and len(rollout_data.get("rollouts", [])) > 0
    has_human = human_data is not None and len(human_data.get("rollouts", [])) > 0
    num_panels = int(has_rollout) + int(has_human) + int(has_rollout and has_human) * 2

    # Use 2x2 grid when we have 4 panels, otherwise single row
    if num_panels == 4:
        fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes_2d.flatten().tolist()
    else:
        fig, ax_raw = plt.subplots(1, num_panels, figsize=(7 * num_panels, 6))
        axes = [ax_raw] if num_panels == 1 else list(ax_raw)

    panel_idx = 0

    # Panel 1: Human demos only
    if has_human:
        ax = axes[panel_idx]
        plot_trajectories_on_ax(ax, human_data["rollouts"], color="steelblue", alpha=0.3, label_prefix="Human demo")
        ax.set_title("Human Demos (Top-Down View)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # Panel 2: Policy rollouts only
    if has_rollout:
        ax = axes[panel_idx]
        successes = [r for r in rollout_data["rollouts"] if r["success"]]
        failures = [r for r in rollout_data["rollouts"] if not r["success"]]
        # Compute shared reference from successful rollouts (or all if none succeeded)
        ref_group = successes if successes else rollout_data["rollouts"]
        pol_ref_start, pol_ref_end = _compute_mean_start_end(ref_group)
        # Successful trajectories: full similarity transform (align start + end)
        if successes:
            plot_trajectories_on_ax(ax, successes, color="seagreen", alpha=0.3,
                                    label_prefix="Policy success",
                                    ref_start=pol_ref_start, ref_end=pol_ref_end, align_end=True)
        # Failed trajectories: translate-only (align start, keep natural end)
        if failures:
            plot_trajectories_on_ax(ax, failures, color="tomato", alpha=0.3,
                                    label_prefix="Policy failure",
                                    ref_start=pol_ref_start, ref_end=pol_ref_end, align_end=False)
        sr = rollout_data.get("success_rate", 0)
        ax.set_title(f"Policy Rollouts (SR={sr:.0%})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # Panel 3: Overlay comparison (use high-contrast colors for distinguishability)
    # All groups are normalized to the human demo's mean start/end as the shared reference
    if has_rollout and has_human:
        ax = axes[panel_idx]
        # Compute human demo reference frame
        ref_start, ref_end = _compute_mean_start_end(human_data["rollouts"])

        # Human demos: full similarity transform
        plot_trajectories_on_ax(ax, human_data["rollouts"], color="royalblue", alpha=0.25,
                                label_prefix="Human demo", ref_start=ref_start, ref_end=ref_end,
                                align_end=True)
        # Policy successes: full similarity transform to same reference
        successes = [r for r in rollout_data["rollouts"] if r["success"]]
        failures = [r for r in rollout_data["rollouts"] if not r["success"]]
        if successes:
            plot_trajectories_on_ax(ax, successes, color="orangered", alpha=0.25,
                                    label_prefix="Policy success", ref_start=ref_start, ref_end=ref_end,
                                    align_end=True)
        # Policy failures: translate-only (align start, keep natural end)
        if failures:
            plot_trajectories_on_ax(ax, failures, color="tomato", alpha=0.25,
                                    label_prefix="Policy failure", ref_start=ref_start, ref_end=ref_end,
                                    align_end=False)
        ax.set_title("Overlay Comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # Sync axis limits across all trajectory panels so they share the same view window
    traj_axes = axes[:panel_idx]
    if len(traj_axes) > 1:
        xlims = [ax.get_xlim() for ax in traj_axes]
        ylims = [ax.get_ylim() for ax in traj_axes]
        shared_xlim = (min(lo for lo, _ in xlims), max(hi for _, hi in xlims))
        shared_ylim = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))
        for ax in traj_axes:
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)

    # Panel 4: Velocity histogram PCA (human vs robot)
    if has_rollout and has_human:
        ax = axes[panel_idx]
        plot_velocity_pca(ax, human_data["rollouts"], rollout_data["rollouts"], method=embed_method)

    # Use a shortened prompt for the figure title
    short_prompt = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    fig.suptitle(short_prompt, fontsize=11, y=1.02)
    fig.tight_layout()

    # Save figure
    safe_name = prompt.replace(" ", "_")[:100]
    fig_path = output_path / f"{safe_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved figure to {fig_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize LIBERO EEF trajectories (top-down view)")
    parser.add_argument("--prompt", type=str, required=True, help="Task prompt text (must match JSON filename)")
    parser.add_argument("--rollout-dir", type=str, default=".cache/output/libero_output_json",
                        help="Directory containing policy rollout JSONs")
    parser.add_argument("--human-dir", type=str, default=".cache/output/libero_output_json/human_demos",
                        help="Directory containing human demo JSONs")
    parser.add_argument("--output-dir", type=str, default=".cache/output/libero_output_json/figures",
                        help="Directory to save output figures")
    parser.add_argument("--embed-method", type=str, default="pca", choices=["pca", "tsne"],
                        help="Dimensionality reduction method for the kinematics panel (default: pca)")
    args = parser.parse_args()
    visualize(args.prompt, args.rollout_dir, args.human_dir, args.output_dir, embed_method=args.embed_method)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
