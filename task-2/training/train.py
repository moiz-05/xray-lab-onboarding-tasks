from pathlib import Path

from mmengine.config import Config
from mmdet.registry import RUNNERS

ROOT = Path(__file__).resolve().parents[1]

config_path = ROOT / 'mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

cfg = Config.fromfile(str(config_path))

# =====================================================
# CUSTOM DATASET SETTINGS
# =====================================================

cfg.train_dataloader.dataset.ann_file = str(ROOT / 'dataset/_annotations.coco.json')
cfg.train_dataloader.dataset.data_prefix.img = str(ROOT / 'dataset/train')
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False

cfg.val_dataloader.dataset.ann_file = str(ROOT / 'dataset/_annotations.coco.json')
cfg.val_dataloader.dataset.data_prefix.img = str(ROOT / 'dataset/train')
cfg.val_dataloader.num_workers = 0
cfg.val_dataloader.persistent_workers = False

cfg.test_dataloader.dataset.ann_file = str(ROOT / 'dataset/_annotations.coco.json')
cfg.test_dataloader.dataset.data_prefix.img = str(ROOT / 'dataset/train')
cfg.test_dataloader.num_workers = 0
cfg.test_dataloader.persistent_workers = False

cfg.val_evaluator.ann_file = str(ROOT / 'dataset/_annotations.coco.json')
cfg.test_evaluator.ann_file = str(ROOT / 'dataset/_annotations.coco.json')

# =====================================================
# CLASSES
# =====================================================

classes = ('avocado', 'apple', 'banana', 'grape')

cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
cfg.test_dataloader.dataset.metainfo = dict(classes=classes)

# =====================================================
# NUM CLASSES
# =====================================================

cfg.model.bbox_head.num_classes = 4
cfg.model.backbone.init_cfg = None

# =====================================================
# CPU TRAINING SETTINGS
# =====================================================

cfg.train_dataloader.batch_size = 1

cfg.train_cfg.max_epochs = 2

cfg.work_dir = str(ROOT / 'outputs')

# =====================================================
# BUILD RUNNER
# =====================================================

runner = RUNNERS.build(cfg)

runner.train()
