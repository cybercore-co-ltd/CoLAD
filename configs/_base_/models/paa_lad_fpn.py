_base_ = './_paa_distil_fpn.py'
model = dict(
    type='LAD',
    bbox_head=dict(type='PAA_LAD_Head'),
    teacher_bbox_head=dict(type='PAA_LAD_Head'))
