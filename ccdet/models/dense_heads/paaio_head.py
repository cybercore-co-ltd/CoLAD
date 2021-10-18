import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmcv.cnn import normal_init, bias_init_with_prob

from mmdet.models import HEADS
from mmdet.core import multi_apply, multiclass_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.dense_heads.paa_head import PAAHead, levels_to_images, EPS


@HEADS.register_module()
class PAAIOHead(PAAHead):
    """The only change is that we add Objness branch to Regress Head. And,
    train combine the objectness branch with classification branch in implicit
    manner,  similar to AutoAssign.
    Therefore, the head will have:
        + Classification Head
        + Regression Head: Box head, IoU Head, Objectness.
    In inference:
        + Classication score = Classication prob * Objectness * IoU score
    """

    def __init__(self, *args, **kwargs):
        assert kwargs['loss_cls']['type'] == 'ProbFocalLoss'
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        # Initialize similar to ATSS/PPA
        super()._init_layers()
        # Add additional Objectness branch
        self.objectness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.obj_bias = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)

    def init_weights(self):
        super().init_weights()
        normal_init(self.objectness, std=0.01)
        bias_cls = bias_init_with_prob(0.0001)
        normal_init(self.obj_bias, std=0.01, bias=bias_cls)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat).float().sigmoid()
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        # We predict iou instead of centerness
        iou_pred = self.atss_centerness(reg_feat)
        obj_score = self.objectness(reg_feat).float().sigmoid()
        obj_bias = self.obj_bias(reg_feat).float().sigmoid()
        cls_score = self.relu(cls_score - obj_bias) * obj_score

        return cls_score, bbox_pred, iou_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when are computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """

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
            label_channels=label_channels,
        )
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]

        # we fuse the cls_scores with obj_scores before feeding to the get_pos_loss
        # cls_obj_scores = [torch.log(s/(1-s)) for s in scores]
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)

        with torch.no_grad():
            labels, label_weights, bbox_weights, num_pos = multi_apply(
                self.paa_reassign,
                pos_losses_list,
                labels,
                labels_weight,
                bboxes_weight,
                pos_inds,
                pos_gt_index,
                anchor_list,
            )
            num_pos = sum(num_pos)
        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        labels = torch.cat(labels, 0).view(-1)
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(labels_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))

        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)

        losses_cls = self.loss_cls(
            cls_scores,
            labels,
            labels_weight,
            avg_factor=max(num_pos, len(img_metas)))  # avoid num_pos=0
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten],
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
                iou_target.clamp(min=EPS),
                avg_factor=iou_target.sum())
        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            iou_preds (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                iou_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        This method is almost same as `ATSSHead._get_bboxes_single()`.
        We use sqrt(iou_preds * cls_scores) in NMS process instead of just
        cls_scores. Besides, score voting is used when `` score_voting``
        is set to True.
        """
        assert with_nms, 'PAA only supports "with_nms=True" now'
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_preds = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_scores, bbox_preds, iou_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * iou_pred[:, None]).sqrt().max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                iou_pred = iou_pred[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_preds.append(iou_pred)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_preds = torch.cat(mlvl_iou_preds)
        mlvl_nms_scores = (mlvl_scores * mlvl_iou_preds[:, None]).sqrt()
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_nms_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=None)
        if self.with_score_voting:
            det_bboxes, det_labels = self.score_voting(det_bboxes, det_labels,
                                                       mlvl_bboxes,
                                                       mlvl_nms_scores,
                                                       cfg.score_thr)

        return det_bboxes, det_labels

    def forward_la(self,
                   x,
                   img_metas,
                   gt_bboxes,
                   gt_labels,
                   gt_bboxes_ignore=None):
        outs = self(x)
        return self.get_la(*outs, gt_bboxes, gt_labels, img_metas,
                           gt_bboxes_ignore=gt_bboxes_ignore)

    def get_la(self,
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
            label_channels=label_channels,
        )
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)

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
        # cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        # bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        # iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))

        return labels, labels_weight, bboxes_target, flatten_anchors, num_pos
