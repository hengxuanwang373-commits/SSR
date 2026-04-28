import argparse
import importlib
import os
import sys


sys.path.append('')


CHECK_KEYWORDS = [
    'img_backbone',
    'img_neck',
    'pts_bbox_head.transformer',
    'pts_bbox_head.tokenlearner',
    'pts_bbox_head.latent_decoder',
    'pts_bbox_head.way_decoder',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check SSR freeze_cfg trainable parameter status')
    parser.add_argument('--config', required=True, help='config file path')
    return parser.parse_args()


def import_plugins(cfg, config_path):
    from mmcv.utils import import_modules_from_strings

    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])

    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)

    module_path = module_dir.replace('/', '.').strip('.')
    if module_path:
        importlib.import_module(module_path)


def apply_freeze_if_available(model):
    if hasattr(model, 'apply_freeze_cfg'):
        model.apply_freeze_cfg()
        return 'apply_freeze_cfg'
    if hasattr(model, '_apply_freeze_cfg'):
        model._apply_freeze_cfg()
        return '_apply_freeze_cfg'
    return None


def main():
    args = parse_args()

    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    applied_method = apply_freeze_if_available(model)
    if applied_method is None:
        print('No apply_freeze_cfg or _apply_freeze_cfg method found.')
    else:
        print(f'Called {applied_method}().')

    total_params = 0
    trainable_params = 0
    keyword_stats = {
        keyword: dict(tensors=0, numel=0)
        for keyword in CHECK_KEYWORDS
    }

    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if not param.requires_grad:
            continue

        trainable_params += numel
        print(f'{name}, shape={tuple(param.shape)}, numel={numel}')

        for keyword, stats in keyword_stats.items():
            if keyword in name:
                stats['tensors'] += 1
                stats['numel'] += numel

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    print('\nParameter summary:')
    print(f'total params: {total_params}')
    print(f'trainable params: {trainable_params}')
    print(f'trainable ratio: {trainable_ratio:.6f}')

    print('\nKeyword trainable parameter summary:')
    for keyword, stats in keyword_stats.items():
        print(
            f'{keyword}: trainable tensors={stats["tensors"]}, '
            f'trainable params={stats["numel"]}')


if __name__ == '__main__':
    main()
