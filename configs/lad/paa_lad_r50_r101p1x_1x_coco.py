_base_ = [
    '../_base_/models/paa_lad_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # student
    pretrained='torchvision://resnet50',
    backbone=dict(depth=50),
    # teacher
    teacher_pretrained="http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth",
    teacher_backbone=dict(depth=101))
# optimizer
optimizer = dict(lr=0.01)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
