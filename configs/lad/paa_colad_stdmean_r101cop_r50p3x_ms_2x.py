_base_ = [
    '../_base_/models/paa_colad_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
load_from = "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth"
model = dict(
    type='CoLAD',
    pretrained='torchvision://resnet50',
    teacher_pretrained='torchvision://resnet101',
    # network 1
    backbone=dict(depth=50),
    bbox_head=dict(type='PAA_CoLAD_StdMean_Head'),
    # network 2
    teacher_backbone=dict(depth=101),
    teacher_bbox_head=dict(
        type='PAA_CoLAD_StdMean_COP_Head',
        loss_cls=dict(
            type='ProbFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # train and test cfg
    test_cfg=dict(use_teacher=True))
# dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# note: total bs = 16, we train CoLAD with 4 GPUS, so each GPU will have 4 samples
data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(lr=0.01)
