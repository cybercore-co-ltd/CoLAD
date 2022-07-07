import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.models import (SingleStageDetector, DETECTORS,
                          build_backbone, build_neck, build_head)


@DETECTORS.register_module()
class CoLAD(SingleStageDetector):
    """Co-Learning Label Assignment Distillation
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
                 # cfgs
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 teacher_pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_bbox_head = build_head(teacher_bbox_head)
        self.init_teacher_weights(teacher_pretrained)

    def init_teacher_weights(self, pretrained):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.teacher_backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.teacher_neck, nn.Sequential):
                for m in self.teacher_neck:
                    m.init_weights()
            else:
                self.teacher_neck.init_weights()
        self.teacher_bbox_head.init_weights()

    def extract_teacher_feat(self, img):
        x = self.teacher_backbone(img)
        if self.with_neck:
            x = self.teacher_neck(x)
        return x

    def extract_teacher_feats(self, imgs):
        assert isinstance(imgs, list)
        return [self.extract_teacher_feat(img) for img in imgs]

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # forward student
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        (pos_losses_list, anchor_list, labels, labels_weight,
            bboxes_target, bboxes_weight, _, pos_inds, pos_gt_index) = \
            self.bbox_head._get_loss_la(*outs,
                                        gt_bboxes,
                                        gt_labels,
                                        img_metas,
                                        gt_bboxes_ignore)

        # fordward teacher
        x_teacher = self.extract_teacher_feat(img)
        outs_teacher = self.teacher_bbox_head(x_teacher)
        t_pos_losses_list = \
            self.teacher_bbox_head._get_loss_la(*outs_teacher,
                                                gt_bboxes,
                                                gt_labels,
                                                img_metas,
                                                gt_bboxes_ignore)[0]

        # perform label assignment for both student and teacher
        (labels, labels_weight, bboxes_target, bboxes_weight,
         pos_inds_flatten, pos_anchors, num_pos, dynamic_statistics) = \
            self.teacher_bbox_head._label_reassign(pos_losses_list,
                                                   t_pos_losses_list,
                                                   anchor_list,
                                                   labels,
                                                   labels_weight,
                                                   bboxes_target,
                                                   bboxes_weight,
                                                   pos_inds,
                                                   pos_gt_index)
        la_results = (labels, labels_weight, bboxes_target, bboxes_weight,
                      pos_inds_flatten, pos_anchors, num_pos)

        # compute student loss based on the assignment results
        losses = self.bbox_head.forward_train_wo_la(
            outs, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore, la_results=la_results)

        # compute teacher loss based on the assignment results
        losses_teacher = self.teacher_bbox_head.forward_train_wo_la(
            outs_teacher, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore, la_results=la_results)

        for key, val in losses_teacher.items():
            losses[key+'_t'] = val

        losses.update(dynamic_statistics)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # use teacher
        if self.test_cfg.get('use_teacher', False):
            x = self.extract_teacher_feat(img)
            outs = self.teacher_bbox_head(x)
            bbox_list = self.teacher_bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels,
                            self.teacher_bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list]

        # use student
        else:
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list]

        return bbox_results
