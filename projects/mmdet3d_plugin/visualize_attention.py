"""
Attention Visualization for SSR Model
Visualizes the 16 token attention weights from TokenLearner
Randomly selects a frame from the test set
"""
import os
import os.path as osp
import random
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


def visualize_attention_maps(attention_weights, save_dir, frame_idx=0):
    """Visualize attention weights for each token as heatmaps."""
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]

    num_tokens = attention_weights.shape[0]
    bev_h, bev_w = 100, 100

    os.makedirs(save_dir, exist_ok=True)

    # Grid visualization - all tokens in one image
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # Square figure
    axes = axes.flatten()

    for token_idx in range(num_tokens):
        attn_map = attention_weights[token_idx].cpu().detach().numpy()
        attn_map = attn_map.reshape(bev_h, bev_w)

        ax = axes[token_idx]
        im = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
        ax.set_title(f'Token {token_idx}', fontsize=12)
        ax.axis('off')

    plt.suptitle(f'Frame {frame_idx} - All Tokens (Grid)', fontsize=16)
    plt.tight_layout()

    save_path = osp.join(save_dir, f'attention_grid_frame_{frame_idx:04d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Individual token visualizations - use same size for all
    for token_idx in range(num_tokens):
        attn_map = attention_weights[token_idx].cpu().detach().numpy()
        attn_map = attn_map.reshape(bev_h, bev_w)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Frame {frame_idx} - Token {token_idx}', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        save_path = osp.join(save_dir, f'attention_token{token_idx}_frame_{frame_idx:04d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    # Also save a single token comparison (token 0) for quick visual check
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    attn_map = attention_weights[0].cpu().detach().numpy().reshape(bev_h, bev_w)
    im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Frame {frame_idx} - Token 0 ONLY', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path = osp.join(save_dir, f'token0_only_frame_{frame_idx:04d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


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
    # Random seed for reproducibility - change to get different frames
    random.seed(12345)  # Changed to get a clearly different frame
    np.random.seed(12345)

    ckpt_path = '/data3_server8/wanghongxuan/SSR/work_dirs/SSR_e2e/raw6epoch/epoch_6.pth'
    save_dir = '/data3_server8/wanghongxuan/SSR/attention_maps/raw6epoch_weight'
    config_path = '/data3_server8/wanghongxuan/SSR/projects/configs/SSR/SSR_e2e.py'

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
    random_idx = random.randint(0, len(dataset) - 1)
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
