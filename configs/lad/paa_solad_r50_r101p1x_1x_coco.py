_base_ = [
    '../_base_/models/paa_lad_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='SoLAD',
    # student
    pretrained='torchvision://resnet50',
    backbone=dict(depth=50),
    # teacher
    teacher_pretrained="http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth",
    teacher_backbone=dict(depth=101),
    kd_bbox_head=dict(
        type='PAA_KD_Head',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        min_iou=0.5,
        use_pred_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        kd_bbox_loss=dict(type='GIoULoss', loss_weight=1.3),
        kd_centerness_loss=dict(
            type='FocalKLLoss',
            use_sigmoid=True,
            gamma=0.5,
            reduction='mean',
            loss_weight=10.0),
        # knowledge distillation
        kd_cls_loss=dict(
            type='FocalKLLoss',
            use_sigmoid=True,
            gamma=0.5,
            reduction='mean',
            loss_weight=2.0)))
# optimizer
optimizer = dict(lr=0.01)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
