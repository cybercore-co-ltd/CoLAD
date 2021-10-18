_base_ = [
    '../_base_/models/paaio_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # without IoU prediction
    bbox_head=dict(type='PAAIO_BASE_Head'))
# optimizer
optimizer = dict(lr=0.01)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
