_base_ = [
    '../_base_/models/paacop_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(depth=18),
    neck=dict(in_channels=[64, 128, 256, 512]))
# optimizer
optimizer = dict(lr=0.01)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
