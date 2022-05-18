import torch
from mmcv.runner import _load_checkpoint
from mmdet.models import (SingleStageDetector, DETECTORS,
                          build_backbone, build_neck, build_head)


@DETECTORS.register_module()
class KnowledgeDistiller(SingleStageDetector):
    """Knowledge Distillation"""

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
                 pretrained=None):
        # build student
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)

        # build teacher
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_bbox_head = build_head(teacher_bbox_head)
        self.init_teacher_weights(teacher_pretrained=teacher_pretrained)

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
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # forward teacher to get outputs
        with torch.no_grad():
            # MUST force teacher to `eval` every training step, b/c at the
            # beginning of epoch, the runner calls all nn.Module elements
            # to be `train`
            self.teacher_eval()

            # forward teacher
            x_teacher = self.extract_teacher_feat(img)
            outs_teacher = self.teacher_bbox_head(x_teacher)

        # student receives the teacher outputs to learn
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(
            x, outs_teacher, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore)

        return losses
