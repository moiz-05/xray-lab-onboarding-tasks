from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

import mmcv
import matplotlib.pyplot as plt

CONFIG = '../configs/fruits_rtmdet_config.py'

CHECKPOINT = '../outputs/latest.pth'

IMAGE = '../dataset/train/sample.jpg'

model = init_detector(
    CONFIG,
    CHECKPOINT,
    device='cpu'
)

result = inference_detector(model, IMAGE)

image = mmcv.imread(IMAGE)
image = mmcv.imconvert(image, 'bgr', 'rgb')

visualizer = DetLocalVisualizer()

visualizer.dataset_meta = model.dataset_meta

visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt=False,
    show=False
)

plt.figure(figsize=(12, 12))
plt.imshow(visualizer.get_image())
plt.axis('off')
plt.show()