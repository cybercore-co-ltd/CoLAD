_base_ = [
    '../_base_/models/paa_kd_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # Student
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101),
    bbox_head=dict(
        kd_bbox_loss=dict(type='GIoULoss', loss_weight=1.3),
        kd_centerness_loss=dict(
            type='FocalKLLoss',
            use_sigmoid=True,
            gamma=0.5,
            reduction='mean',
            loss_weight=10.0),
        kd_cls_loss=dict(
            type='FocalKLLoss',
            use_sigmoid=True,
            gamma=0.5,
            reduction='mean',
            loss_weight=10.0)),
    # Teacher
    teacher_pretrained="http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth",
    teacher_backbone=dict(depth=50))
# optimizer
optimizer = dict(lr=0.01)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
