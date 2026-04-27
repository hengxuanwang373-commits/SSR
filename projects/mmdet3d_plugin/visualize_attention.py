"""
Attention Visualization for SSR Model
Visualizes the 16 token attention weights from TokenLearner
Randomly selects a frame from the test set
"""
import os
import os.path as osp
import argparse
import random
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/data3_server8/wanghongxuan/SSR')
import sys
sys.path.insert(0, '/data3_server8/wanghongxuan/SSR')

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset

# Import SSR to register the model
from projects.mmdet3d_plugin.SSR.SSR import SSR


def _to_numpy_attention(attention_weights):
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]
    return attention_weights.cpu().detach().float().numpy()


def _normalize_map(attn_map, eps=1e-12):
    min_val = float(attn_map.min())
    max_val = float(attn_map.max())
    return (attn_map - min_val) / (max_val - min_val + eps)


def _save_heatmap(attn_map, save_path, title, vmin=None, vmax=None, show_colorbar=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def _save_grid(attention_maps, save_path, title, vmin=None, vmax=None):
    num_tokens = attention_maps.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for token_idx in range(num_tokens):
        ax = axes[token_idx]
        ax.imshow(attention_maps[token_idx], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'Token {token_idx}', fontsize=12)
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_attention_metrics(attention_weights, topk_percents=(1, 5, 10), eps=1e-12):
    """Compute analysis metrics for TokenLearner spatial attention maps."""
    attention = _to_numpy_attention(attention_weights)
    num_tokens, num_cells = attention.shape

    attn = attention / (attention.sum(axis=1, keepdims=True) + eps)
    entropy = -(attn * np.log(attn + eps)).sum(axis=1)
    entropy_norm = entropy / np.log(num_cells)

    metrics = {
        'num_tokens': int(num_tokens),
        'num_cells': int(num_cells),
        'entropy': {
            'per_token': entropy.tolist(),
            'normalized_per_token': entropy_norm.tolist(),
            'mean': float(entropy.mean()),
            'std': float(entropy.std()),
            'normalized_mean': float(entropy_norm.mean()),
            'normalized_std': float(entropy_norm.std()),
        },
        'topk_mass': {},
        'token_diversity': {},
        'token_overlap': {},
    }

    flat_norm = np.linalg.norm(attn, axis=1, keepdims=True)
    normed = attn / (flat_norm + eps)
    cosine_sim = normed @ normed.T
    cosine_distance = 1.0 - cosine_sim
    pair_mask = ~np.eye(num_tokens, dtype=bool)
    pairwise_cosine_distance = cosine_distance[pair_mask]
    metrics['token_diversity'] = {
        'pairwise_cosine_distance_mean': float(pairwise_cosine_distance.mean()),
        'pairwise_cosine_distance_std': float(pairwise_cosine_distance.std()),
        'pairwise_cosine_distance_min': float(pairwise_cosine_distance.min()),
        'pairwise_cosine_distance_max': float(pairwise_cosine_distance.max()),
    }

    sorted_attn = np.sort(attn, axis=1)[:, ::-1]
    for percent in topk_percents:
        k = max(1, int(np.ceil(num_cells * percent / 100.0)))
        masses = sorted_attn[:, :k].sum(axis=1)
        metrics['topk_mass'][f'top_{percent}_percent'] = {
            'k': int(k),
            'per_token': masses.tolist(),
            'mean': float(masses.mean()),
            'std': float(masses.std()),
            'min': float(masses.min()),
            'max': float(masses.max()),
        }

        top_indices = np.argpartition(attn, -k, axis=1)[:, -k:]
        masks = np.zeros((num_tokens, num_cells), dtype=bool)
        rows = np.arange(num_tokens)[:, None]
        masks[rows, top_indices] = True

        overlaps = []
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                overlaps.append(float(intersection / (union + eps)))

        overlaps = np.asarray(overlaps, dtype=np.float64)
        metrics['token_overlap'][f'top_{percent}_percent_jaccard'] = {
            'mean': float(overlaps.mean()),
            'std': float(overlaps.std()),
            'min': float(overlaps.min()),
            'max': float(overlaps.max()),
        }

    return metrics


def visualize_attention_maps(attention_weights, save_dir, frame_idx=0, topk_percents=(1, 5, 10)):
    """Visualize TokenLearner spatial attention maps and save analysis metrics."""
    attention_weights_np = _to_numpy_attention(attention_weights)

    num_tokens = attention_weights_np.shape[0]
    bev_h, bev_w = 100, 100
    attention_maps = attention_weights_np.reshape(num_tokens, bev_h, bev_w)

    os.makedirs(save_dir, exist_ok=True)
    current_style_dir = osp.join(save_dir, 'raw_current_style')
    per_token_dir = osp.join(save_dir, 'per_token_norm')
    global_dir = osp.join(save_dir, 'global_norm')
    topk_dir = osp.join(save_dir, 'topk_masks')
    raw_dir = osp.join(save_dir, 'raw')
    for path in [current_style_dir, per_token_dir, global_dir, topk_dir, raw_dir]:
        os.makedirs(path, exist_ok=True)

    np.save(osp.join(raw_dir, f'attention_weights_frame_{frame_idx:04d}.npy'), attention_weights_np)
    torch.save(
        torch.as_tensor(attention_weights_np),
        osp.join(raw_dir, f'attention_weights_frame_{frame_idx:04d}.pt'))

    metrics = compute_attention_metrics(attention_weights, topk_percents=topk_percents)
    metrics_path = osp.join(save_dir, f'attention_metrics_frame_{frame_idx:04d}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Current style: keep the previous implicit matplotlib scaling for backward comparison.
    _save_grid(
        attention_maps,
        osp.join(current_style_dir, f'attention_grid_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - All Tokens (Current Style)')
    _save_grid(
        attention_maps,
        osp.join(save_dir, f'attention_grid_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - All Tokens (Current Style)')
    print(f"Saved: {osp.join(save_dir, f'attention_grid_frame_{frame_idx:04d}.png')}")

    for token_idx in range(num_tokens):
        attn_map = attention_maps[token_idx]
        _save_heatmap(
            attn_map,
            osp.join(current_style_dir, f'attention_token{token_idx}_frame_{frame_idx:04d}.png'),
            f'Frame {frame_idx} - Token {token_idx}')
        _save_heatmap(
            attn_map,
            osp.join(save_dir, f'attention_token{token_idx}_frame_{frame_idx:04d}.png'),
            f'Frame {frame_idx} - Token {token_idx}')

    _save_heatmap(
        attention_maps[0],
        osp.join(current_style_dir, f'token0_only_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - Token 0 ONLY')
    _save_heatmap(
        attention_maps[0],
        osp.join(save_dir, f'token0_only_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - Token 0 ONLY')

    # Per-token normalization: each token has its own [0, 1] color scale.
    per_token_maps = np.stack([_normalize_map(attention_maps[i]) for i in range(num_tokens)], axis=0)
    _save_grid(
        per_token_maps,
        osp.join(per_token_dir, f'attention_grid_per_token_norm_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - Per-Token Normalization',
        vmin=0,
        vmax=1)

    for token_idx in range(num_tokens):
        _save_heatmap(
            per_token_maps[token_idx],
            osp.join(per_token_dir, f'attention_token{token_idx}_per_token_norm_frame_{frame_idx:04d}.png'),
            f'Frame {frame_idx} - Token {token_idx} (Per-Token Norm)',
            vmin=0,
            vmax=1)

    # Global normalization: all tokens share one color scale.
    global_min = float(attention_maps.min())
    global_max = float(attention_maps.max())
    _save_grid(
        attention_maps,
        osp.join(global_dir, f'attention_grid_global_norm_frame_{frame_idx:04d}.png'),
        f'Frame {frame_idx} - Global Normalization',
        vmin=global_min,
        vmax=global_max)

    for token_idx in range(num_tokens):
        _save_heatmap(
            attention_maps[token_idx],
            osp.join(global_dir, f'attention_token{token_idx}_global_norm_frame_{frame_idx:04d}.png'),
            f'Frame {frame_idx} - Token {token_idx} (Global Norm)',
            vmin=global_min,
            vmax=global_max)

    # Top-k masks: binary maps for the most attended BEV cells.
    flat = attention_weights_np.reshape(num_tokens, -1)
    num_cells = flat.shape[1]
    for percent in topk_percents:
        percent_dir = osp.join(topk_dir, f'top_{percent}_percent')
        os.makedirs(percent_dir, exist_ok=True)
        k = max(1, int(np.ceil(num_cells * percent / 100.0)))
        top_indices = np.argpartition(flat, -k, axis=1)[:, -k:]
        masks = np.zeros_like(flat, dtype=np.float32)
        rows = np.arange(num_tokens)[:, None]
        masks[rows, top_indices] = 1.0
        mask_maps = masks.reshape(num_tokens, bev_h, bev_w)

        _save_grid(
            mask_maps,
            osp.join(percent_dir, f'attention_grid_top_{percent}_percent_frame_{frame_idx:04d}.png'),
            f'Frame {frame_idx} - Top {percent}% Masks',
            vmin=0,
            vmax=1)
        for token_idx in range(num_tokens):
            _save_heatmap(
                mask_maps[token_idx],
                osp.join(percent_dir, f'attention_token{token_idx}_top_{percent}_percent_frame_{frame_idx:04d}.png'),
                f'Frame {frame_idx} - Token {token_idx} Top {percent}%',
                vmin=0,
                vmax=1,
                show_colorbar=False)

    print(f"Saved: {osp.join(save_dir, f'token0_only_frame_{frame_idx:04d}.png')}")
    print(f"Saved metrics: {metrics_path}")


class AttentionVisualizer:
    """Hook-based attention extractor for SSR model."""

    def __init__(self, model, save_dir):
        self.save_dir = save_dir
        self.attention_weights = None
        self.frame_idx = 0
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register forward hooks to extract attention weights."""
        def hook_fn(module, input, output):
            self.attention_weights = output[1].detach().cpu()
            print(f"Captured attention weights shape: {self.attention_weights.shape}")

        if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'tokenlearner'):
            target = model.pts_bbox_head.tokenlearner
            handle = target.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            print(f"Registered hook on TokenLearner")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description='Visualize SSR TokenLearner attention heatmaps.')
    parser.add_argument('--checkpoint', default='/data3_server8/wanghongxuan/SSR/work_dirs/SSR_mini_cbam_6epoch/epoch_6.pth')
    parser.add_argument('--config', default='/data3_server8/wanghongxuan/SSR/projects/configs/SSR/SSR_e2e_mini.py')
    parser.add_argument('--save-dir', default='/data3_server8/wanghongxuan/SSR/attention_maps/epoch_6_mini_cbam')
    parser.add_argument('--frame-idx', type=int, default=None, help='Dataset index to visualize. Random if omitted.')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # Random seed for reproducibility - change to get different frames
    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.checkpoint
    save_dir = args.save_dir
    config_path = args.config

    os.makedirs(save_dir, exist_ok=True)

    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.cuda()

    visualizer = AttentionVisualizer(model, save_dir)

    # Build dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    print(f"Dataset size: {len(dataset)}")

    # Select random index
    random_idx = args.frame_idx if args.frame_idx is not None else random.randint(0, len(dataset) - 1)
    if random_idx < 0 or random_idx >= len(dataset):
        raise ValueError(f'frame_idx {random_idx} out of range [0, {len(dataset) - 1}]')
    print(f"Random selected frame index: {random_idx}")

    # Create subset with just the random index
    subset = Subset(dataset, [random_idx])

    # Build dataloader for subset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(subset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    print("Processing random frame...")

    for data in data_loader:
        # Unwrap data from DataContainer format and move to GPU
        img = data['img'][0].data[0].cuda()  # [B, N, C, H, W]
        img_metas = data['img_metas'][0].data[0]  # list of dicts (no cuda needed)
        ego_his_trajs = data['ego_his_trajs'][0].data[0].cuda()
        ego_lcf_feat = data['ego_lcf_feat'][0].data[0].cuda()
        cmd = data['ego_fut_cmd'][0].data[0].cuda()

        print(f"Image shape: {img.shape}")
        print(f"img_metas type: {type(img_metas)}, len: {len(img_metas)}")

        # Get command info
        cmd_list = ['Go Straight', 'Turn Left', 'Turn Right']
        cmd_idx = cmd.argmax().item()
        print(f"Command: {cmd_list[cmd_idx]}")

        with torch.no_grad():
            mlvl_feats = model.extract_feat(img=img, img_metas=img_metas)

            prev_bev = None
            bev_embed = model.pts_bbox_head(
                mlvl_feats, img_metas, prev_bev=prev_bev,
                ego_his_trajs=ego_his_trajs,
                ego_lcf_feat=ego_lcf_feat,
                cmd=cmd
            )

        # Save with the random index
        visualize_attention_maps(visualizer.attention_weights, save_dir, frame_idx=random_idx)
        visualizer.remove_hooks()
        break

    print(f"\nDone! Visualization saved to: {save_dir}")
    print(f"Frame index: {random_idx}")


if __name__ == '__main__':
    main()
