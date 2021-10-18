import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32

from mmdet.core import multiclass_nms
from mmdet.models import HEADS, PAAHead


@HEADS.register_module()
class PAA_CoLAD_StdMean_Head(PAAHead):
    """PAA Head for Co-learning Label Assignment Distillation (CoLAD)
    with Std/Mean criterion https://arxiv.org/abs/2108.10520"""


@HEADS.register_module()
class PAA_CoLAD_StdMean_COP_Head(PAA_CoLAD_StdMean_Head):
    """Co-Learning COP Head with Std/Mean criterion"""

    def __init__(self, coeff_kernel_size=3, *args, **kwargs):
        assert kwargs['loss_cls']['type'] == 'ProbFocalLoss'
        self.coeff_kernel_size = coeff_kernel_size
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        # Initialize similar to ATSS/PPA
        super()._init_layers()
        # Add additional Objectness branch
        self.objectness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)

        # Local IO
        self.padding = (self.coeff_kernel_size-1)//2
        # Add additional Objectness branch
        self.coeff_conv = nn.Conv2d(
            self.feat_channels, self.coeff_kernel_size**2, 3, padding=1)

        self.unfold = nn.Unfold(
            [self.coeff_kernel_size, self.coeff_kernel_size],
            padding=self.padding)

    def init_weights(self):
        super().init_weights()
        normal_init(self.objectness, std=0.01)
        normal_init(self.coeff_conv, std=0.01)

    def forward_single(self, x, scale):
        assert self.num_anchors == 1, "only support num anchors = 1"
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        # We predict iou instead of centerness
        iou_pred = self.atss_centerness(reg_feat)

        # classification
        N, C, H, W = cls_score.shape
        cls_score = cls_score.view(N*C, 1, H, W)
        cls_score = self.unfold(cls_score)  # ((N*C)x(k*k)x(H*W)
        cls_score = cls_score.permute(0, 2, 1).view(
            N, C, H, W, self.coeff_kernel_size**2)  # NxCxHxWx(k*k)

        # coeffs
        coeffs = self.coeff_conv(reg_feat)  # Nx(k*k)xHxW
        coeffs = coeffs.permute(0, 2, 3, 1).view(
            N, 1, H, W, self.coeff_kernel_size**2)  # Nx1xHxWx(k*k)

        # fusion
        cls_score = cls_score.sigmoid()
        coeffs = coeffs.sigmoid()
        cls_score = (cls_score * coeffs).mean(-1)

        return cls_score, bbox_pred, iou_pred

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
