import torch
from mmcv.runner import force_fp32

from mmdet.models import HEADS, PAAHead
from mmdet.core import multi_apply, bbox_overlaps
from mmdet.models.dense_heads.paa_head import levels_to_images


@HEADS.register_module()
class PAA_LAD_Head(PAAHead):
    """PAA head for Label Assignment Distillation
    https://arxiv.org/abs/2108.10520"""

    def forward_la(self,
                   x,
                   img_metas,
                   gt_bboxes,
                   gt_labels,
                   gt_bboxes_ignore=None,
                   return_outputs=False):
        outs = self(x)
        (pos_losses_list, anchor_list, labels, labels_weight, bboxes_target,
         bboxes_weight, bbox_preds, pos_inds, pos_gt_index) = \
            self._get_loss_la(*outs,
                              gt_bboxes,
                              gt_labels,
                              img_metas,
                              gt_bboxes_ignore)
        label_reassign = self._label_reassign(pos_losses_list,
                                              anchor_list,
                                              labels,
                                              labels_weight,
                                              bboxes_target,
                                              bboxes_weight,
                                              bbox_preds,
                                              pos_inds,
                                              pos_gt_index)
        if return_outputs:
            return (label_reassign, outs)
        else:
            return label_reassign

    def _get_loss_la(self,
                     cls_scores,
                     bbox_preds,
                     iou_preds,
                     gt_bboxes,
                     gt_labels,
                     img_metas,
                     gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)
        return (pos_losses_list, anchor_list, labels, labels_weight,
                bboxes_target, bboxes_weight, bbox_preds, pos_inds, pos_gt_index)

    def _label_reassign(self,
                        pos_losses_list,
                        anchor_list,
                        labels,
                        labels_weight,
                        bboxes_target,
                        bboxes_weight,
                        bbox_preds,
                        pos_inds,
                        pos_gt_index):
        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
                reassign_bbox_weights, num_pos = multi_apply(
                    self.paa_reassign,
                    pos_losses_list,
                    labels,
                    labels_weight,
                    bboxes_weight,
                    pos_inds,
                    pos_gt_index,
                    anchor_list)
            num_pos = sum(num_pos)
        # convert all tensor list to a flatten tensor
        labels = torch.cat(reassign_labels, 0).view(-1)
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))
        flatten_anchors = torch.cat([torch.cat(item, 0)
                                     for item in anchor_list])
        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)

        if num_pos:
            pos_anchors = flatten_anchors[pos_inds_flatten]
        else:
            pos_anchors = None
        return (labels, labels_weight, bboxes_target, bboxes_weight,
                pos_inds_flatten, pos_anchors, num_pos)

    def forward_train_wo_la(self,
                            outs,
                            img_metas,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore,
                            la_results,
                            proposal_cfg=None):
        # The rest is indentical to that of Parent Class
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss_wo_la(*loss_inputs,
                                 gt_bboxes_ignore=gt_bboxes_ignore,
                                 la_results=la_results)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss_wo_la(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   gt_bboxes,
                   gt_labels,
                   img_metas,
                   gt_bboxes_ignore=None,
                   la_results=None):

        (labels, labels_weight, bboxes_target, bboxes_weight,
            pos_inds_flatten, pos_anchors, num_pos) = la_results

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, self.cls_out_channels)
                      for item in cls_scores]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))

        losses_cls = self.loss_cls(
            cls_scores,
            labels,
            labels_weight,
            avg_factor=max(num_pos, len(img_metas)))  # avoid num_pos=0
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(
                pos_anchors,
                bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            iou_target = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            losses_iou = self.loss_centerness(
                iou_preds[pos_inds_flatten],
                iou_target.unsqueeze(-1),
                avg_factor=num_pos)
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_target,
                avg_factor=num_pos)

        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)
