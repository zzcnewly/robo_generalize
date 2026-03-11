"""
Action space analysis for MolmoSpaces + DreamZero evaluation trajectories.

Loads HDF5 trajectory files from MolmoSpaces eval_output, extracts action
distributions, and produces UMAP visualizations colored by success/failure.

Usage:
    python action_space_analysis_molmospaces.py \
        --eval_dir eval_output/molmospaces \
        [--action_type ee_twist] \
        [--output_prefix molmospaces_action_analysis]

Action types:
    joint_pos     : 7D absolute joint positions (what DreamZero DROID outputs)
    joint_pos_rel : 7D relative joint position changes
    ee_twist      : 6D Cartesian velocity (dx, dy, dz, wx, wy, wz)
    ee_pose       : 7D end-effector pose (xyz + quaternion)
    tcp_pose      : 7D TCP trajectory from obs (xyz + quaternion), direct float32
"""

import argparse
import glob
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def decode_byte_array(data):
    """Decode a MolmoSpaces uint8 byte array back to a Python object."""
    raw_bytes = bytes(data).rstrip(b'\x00')
    return json.loads(raw_bytes.decode('utf-8'))


def load_trajectories(eval_dir, action_type="ee_twist"):
    """
    Load action trajectories from all HDF5 files under eval_dir.

    Returns:
        trajs: list of (T, D) numpy arrays, one per episode
        metadata: list of dicts with keys {house, traj_idx, success, h5_path}
    """
    h5_files = sorted(glob.glob(os.path.join(eval_dir, '**/trajectories*.h5'), recursive=True))
    if not h5_files:
        raise FileNotFoundError("No HDF5 trajectory files found under %s" % eval_dir)

    print("Found %d HDF5 files" % len(h5_files))
    trajs = []
    metadata = []

    for h5_path in h5_files:
        house = os.path.basename(os.path.dirname(h5_path))
        print("  Loading %s (%s)" % (h5_path, house))

        with h5py.File(h5_path, 'r') as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])

            for traj_key in traj_keys:
                traj_grp = f[traj_key]
                T = traj_grp['success'].shape[0]
                success = bool(traj_grp['success'][-1])

                if action_type == "tcp_pose":
                    actions = traj_grp['obs']['extra']['tcp_pose'][:]
                else:
                    action_key = action_type
                    raw_data = traj_grp['actions'][action_key]

                    action_list = []
                    for t in range(T):
                        decoded = decode_byte_array(raw_data[t])
                        if isinstance(decoded, dict):
                            arm = np.array(decoded['arm'], dtype=np.float64)
                            action_list.append(arm)
                        else:
                            action_list.append(np.array(decoded, dtype=np.float64))
                    actions = np.array(action_list)

                trajs.append(actions)
                metadata.append({
                    'house': house,
                    'traj_idx': traj_key,
                    'success': success,
                    'h5_path': h5_path,
                    'timesteps': T,
                })

    print("Loaded %d episodes total" % len(trajs))
    return trajs, metadata


# ---------------------------------------------------------------------------
# Feature extraction (generalized from action_space_reduc.py to arbitrary D)
# ---------------------------------------------------------------------------

def extract_trajectory_features(data):
    """
    Extract statistical features from trajectory data.

    Args:
        data: list of N arrays, each (T, D)
    Returns:
        (N, 13*D) feature matrix
    """
    features = []
    for traj in data:
        feat = []
        T, D = traj.shape

        feat.append(traj[0])
        feat.append(traj[-1])
        feat.append(traj[T // 2])

        feat.append(traj.mean(axis=0))
        feat.append(traj.std(axis=0))

        vel = np.diff(traj, axis=0)
        feat.append(vel.mean(axis=0))
        feat.append(vel.std(axis=0))
        feat.append(np.abs(vel).max(axis=0))

        acc = np.diff(vel, axis=0)
        feat.append(acc.mean(axis=0))
        feat.append(acc.std(axis=0))

        total_dist = np.abs(vel).sum(axis=0)
        feat.append(total_dist)

        displacement = traj[-1] - traj[0]
        feat.append(displacement)

        feat.append(np.percentile(traj, 25, axis=0))

        features.append(np.concatenate(feat))
    return np.array(features)


def extract_action_features(data):
    """
    Extract statistical features from action sequences.

    Args:
        data: list of N arrays, each (T, D)
    Returns:
        (N, ~10*D+4) feature matrix
    """
    features = []
    for act in data:
        feat = []
        T, D = act.shape

        feat.append(act.mean(axis=0))
        feat.append(act.std(axis=0))

        magnitudes = np.linalg.norm(act, axis=1)
        feat.append([magnitudes.mean(), magnitudes.std(),
                     magnitudes.max(), magnitudes.min()])

        feat.append(act.sum(axis=0))

        jerk = np.diff(act, axis=0)
        feat.append(jerk.std(axis=0))

        third = max(T // 3, 1)
        feat.append(act[:third].mean(axis=0))
        feat.append(act[third:2*third].mean(axis=0))
        feat.append(act[2*third:].mean(axis=0))

        feat.append(np.percentile(act, 10, axis=0))
        feat.append(np.percentile(act, 90, axis=0))

        features.append(np.concatenate(feat))
    return np.array(features)


def dim_reduction_umap(features, n_components=2, n_neighbors=8, min_dist=0.1):
    """UMAP dimensionality reduction on a feature matrix."""
    import umap
    feat_scaled = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(feat_scaled)


def dim_reduction_pca(features, n_components=2):
    """PCA dimensionality reduction (fallback if umap not available)."""
    from sklearn.decomposition import PCA
    feat_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(feat_scaled)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embeddings(embedding, metadata, title, output_path):
    """Scatter plot of 2D embeddings, colored by success/failure and house."""
    successes = np.array([m['success'] for m in metadata])
    houses = [m['house'] for m in metadata]
    unique_houses = sorted(set(houses))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    fail_mask = ~successes
    ax.scatter(embedding[fail_mask, 0], embedding[fail_mask, 1],
               c='red', marker='x', s=60, alpha=0.7, label='Fail (%d)' % fail_mask.sum())
    ax.scatter(embedding[successes, 0], embedding[successes, 1],
               c='green', marker='o', s=60, alpha=0.7, label='Success (%d)' % successes.sum())
    ax.set_title('%s - Success vs Fail' % title)
    ax.legend()
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

    ax = axes[1]
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(unique_houses), 1))
    house_to_idx = {h: i for i, h in enumerate(unique_houses)}
    for house in unique_houses:
        mask = np.array([h == house for h in houses])
        color = cmap(house_to_idx[house])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color], marker='o', s=60, alpha=0.7, label=house)
    ax.set_title('%s - By House/Scene' % title)
    if len(unique_houses) <= 15:
        ax.legend(fontsize=7, loc='best')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Saved plot: %s" % output_path)
    plt.close()


def plot_action_distributions(trajs, metadata, output_path):
    """Plot per-dimension action distributions (histograms) split by success/fail."""
    successes = np.array([m['success'] for m in metadata])
    D = trajs[0].shape[1]

    all_success = np.concatenate(
        [trajs[i] for i in range(len(trajs)) if successes[i]], axis=0
    ) if successes.any() else np.empty((0, D))
    all_fail = np.concatenate(
        [trajs[i] for i in range(len(trajs)) if not successes[i]], axis=0
    ) if (~successes).any() else np.empty((0, D))

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for d in range(D):
        ax = axes[d // ncols, d % ncols]
        if all_success.shape[0] > 0:
            ax.hist(all_success[:, d], bins=50, alpha=0.5, color='green',
                    label='Success', density=True)
        if all_fail.shape[0] > 0:
            ax.hist(all_fail[:, d], bins=50, alpha=0.5, color='red',
                    label='Fail', density=True)
        ax.set_title('Dim %d' % d, fontsize=9)
        ax.legend(fontsize=7)

    for d in range(D, nrows * ncols):
        axes[d // ncols, d % ncols].set_visible(False)

    plt.suptitle('Per-Dimension Action Distributions', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Saved plot: %s" % output_path)
    plt.close()


def plot_temporal_profiles(trajs, metadata, output_path, max_episodes=30):
    """Plot action magnitude over time for each episode, colored by success."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (traj, meta) in enumerate(zip(trajs[:max_episodes], metadata[:max_episodes])):
        mag = np.linalg.norm(traj, axis=1)
        color = 'green' if meta['success'] else 'red'
        ax.plot(mag, color=color, alpha=0.6, linewidth=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', label='Success'),
        Line2D([0], [0], color='red', label='Fail'),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action magnitude (L2)')
    ax.set_title('Action Magnitude Over Time')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Saved plot: %s" % output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Action space analysis for MolmoSpaces + DreamZero")
    parser.add_argument('--eval_dir', type=str, default='eval_output/molmospaces',
                        help='Root directory containing MolmoSpaces eval output HDF5 files')
    parser.add_argument('--action_type', type=str, default='ee_twist',
                        choices=['joint_pos', 'joint_pos_rel', 'ee_twist',
                                 'ee_pose', 'tcp_pose'],
                        help='Which action representation to analyze')
    parser.add_argument('--output_prefix', type=str,
                        default='molmospaces_action_analysis',
                        help='Prefix for output plot files')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Truncate trajectories to this length (None=full)')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA instead of UMAP for dim reduction')
    parser.add_argument('--feature_type', type=str, default='action',
                        choices=['action', 'trajectory'],
                        help='Feature extraction: action (delta-focused) or '
                             'trajectory (position-focused)')
    args = parser.parse_args()

    print("=" * 60)
    print("MolmoSpaces Action Space Analysis")
    print("  eval_dir:     %s" % args.eval_dir)
    print("  action_type:  %s" % args.action_type)
    print("  feature_type: %s" % args.feature_type)
    print("  reduction:    %s" % ("PCA" if args.use_pca else "UMAP"))
    print("=" * 60)

    trajs, metadata = load_trajectories(args.eval_dir,
                                        action_type=args.action_type)

    if len(trajs) == 0:
        print("No trajectories loaded, exiting.")
        return

    n_success = sum(1 for m in metadata if m['success'])
    n_fail = len(metadata) - n_success
    dims = trajs[0].shape[1]
    lengths = [t.shape[0] for t in trajs]
    print("\nSummary:")
    print("  Episodes:    %d (success=%d, fail=%d)" % (
        len(trajs), n_success, n_fail))
    print("  Action dims: %d" % dims)
    print("  Traj lengths: min=%d, max=%d, mean=%.1f" % (
        min(lengths), max(lengths), np.mean(lengths)))

    if args.max_len is not None:
        trajs = [t[:args.max_len] for t in trajs]
        print("  Truncated to max_len=%d" % args.max_len)

    print("\nExtracting features (%s)..." % args.feature_type)
    if args.feature_type == 'action':
        features = extract_action_features(trajs)
    else:
        features = extract_trajectory_features(trajs)
    print("  Feature matrix: %s" % (features.shape,))

    print("Running %s..." % ("PCA" if args.use_pca else "UMAP"))
    if args.use_pca:
        embedding = dim_reduction_pca(features, n_components=2)
    else:
        embedding = dim_reduction_umap(features, n_components=2)
    print("  Embedding: %s" % (embedding.shape,))

    output_dir = os.path.join(
        os.path.dirname(args.eval_dir) or '.', 'analysis_plots')
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.join(
        output_dir, '%s_%s' % (args.output_prefix, args.action_type))

    plot_embeddings(
        embedding, metadata,
        title='%s (%s)' % (args.action_type, args.feature_type),
        output_path='%s_umap.png' % prefix,
    )

    plot_action_distributions(
        trajs, metadata,
        output_path='%s_distributions.png' % prefix,
    )

    plot_temporal_profiles(
        trajs, metadata,
        output_path='%s_temporal.png' % prefix,
    )

    print("\nDone! All plots saved to %s/" % output_dir)


if __name__ == '__main__':
    main()
