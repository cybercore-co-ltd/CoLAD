from mmcv.runner import _load_checkpoint
from mmdet.models import (SingleStageDetector, DETECTORS,
                          build_backbone, build_neck, build_head)


@DETECTORS.register_module()
class SoLAD(SingleStageDetector):
    """Label Assignment Distillation + SoftLabel
    https://arxiv.org/abs/2108.10520"""

    def __init__(self,
                 # student
                 backbone,
                 neck,
                 bbox_head,
                 # teacher
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_pretrained,
                 # cfgs
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 kd_bbox_head=None):
        assert kd_bbox_head is not None
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        assert kd_bbox_head is not None
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_bbox_head = build_head(teacher_bbox_head)
        self.init_teacher_weights(teacher_pretrained=teacher_pretrained)
        self.kd_bbox_head = build_head(kd_bbox_head)

    def init_teacher_weights(self, teacher_pretrained):
        ckpt = _load_checkpoint(teacher_pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        teacher_ckpt = dict()
        for key in ckpt:
            teacher_ckpt['teacher_' + key] = ckpt[key]
        self.load_state_dict(teacher_ckpt, strict=False)
        print("Init teacher weights done")

    def teacher_eval(self):
        self.teacher_backbone.eval()
        self.teacher_neck.eval()
        self.teacher_bbox_head.eval()

    def student_eval(self):
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

    def student_train(self):
        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()

    def extract_teacher_feat(self, img):
        x = self.teacher_backbone(img)
        if self.with_neck:
            x = self.teacher_neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        raise NotImplementedError
