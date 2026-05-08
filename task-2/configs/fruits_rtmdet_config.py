_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

dataset_type = 'CocoDataset'

classes = ('avocado', 'apple', 'banana', 'grape')

data_root = '../dataset/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file=data_root + '_annotations.coco.json',
        data_prefix=dict(img='train/')
    )
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

val_evaluator = dict(
    ann_file=data_root + '_annotations.coco.json'
)

test_evaluator = val_evaluator

model = dict(
    bbox_head=dict(num_classes=4)
)

train_cfg = dict(max_epochs=2)

work_dir = '../outputs'