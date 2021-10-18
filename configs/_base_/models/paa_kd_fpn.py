_base_ = './_paa_distil_fpn.py'
model = dict(
    type='KnowledgeDistiller',
    #  -------------------------Student network------------------------------
    # need to add pretrain and depth of backbone
    bbox_head=dict(
        type='PAA_KD_Head',
        # Need to add Distill loss for cls and regression
    ),
    # -------------------------Teacher network------------------------------
    # need to add teacher pretrain and depth of backbone
    teacher_bbox_head=dict(type='PAAHead'))
