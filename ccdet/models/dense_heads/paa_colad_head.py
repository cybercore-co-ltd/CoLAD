import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32

from mmdet.models import HEADS, PAAHead
from mmdet.models.dense_heads.paa_head import levels_to_images, EPS, skm
from mmdet.core import (multi_apply, bbox_overlaps, multiclass_nms,
                        bbox_mapping_back, bbox2result)

import numpy as np
from inspect import signature

EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@HEADS.register_module()
class StandardizedPAAHead(PAAHead):
    """The same as PAAHead, excepting code is breaked into functions so that
    other classes can inherit from this base class without rewriting long code.

    Function name convention:
        Breaked functions are named with `_` prefix.
    """

    def __init__(self, norm_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_loss = norm_loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """This loss function is breaked into small functions"""
        # get anchor
        anchor_list, valid_flag_list = self._get_anchors(cls_scores, img_metas)

        # perform pre-assignment
        preassign_targets = self._preassign(gt_bboxes, gt_labels,
                                            gt_bboxes_ignore, img_metas,
                                            anchor_list, valid_flag_list)
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = preassign_targets

        # format preds before computing pos loss
        cls_scores, bbox_preds, iou_preds = \
            self._format_preds_before_preassign(cls_scores, bbox_preds,
                                                iou_preds)

        # compute pos loss
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)

        # perform re-assignment
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

        # format preds before computing loss
        (cls_scores, bbox_preds, iou_preds, flatten_anchors,
         labels, labels_weight, bboxes_target) = \
            self._format_preds_before_loss(cls_scores, bbox_preds, iou_preds,
                                           reassign_labels,
                                           reassign_label_weight,
                                           bboxes_target, anchor_list)

        # compute losses
        losses = self._compute_loss(
            cls_scores, bbox_preds, iou_preds, flatten_anchors,
            labels, labels_weight, bboxes_target, num_pos, img_metas)
        return losses

    def _get_anchors(self, cls_scores, img_metas):
        """get anchors for all imgs in batch"""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes,
                                                        img_metas,
                                                        device=device)
        return anchor_list, valid_flag_list

    def _format_preds_before_preassign(self, cls_scores, bbox_preds, iou_preds):
        """Format preds before pre-assignment"""
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, self.cls_out_channels)
                      for item in cls_scores]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        return cls_scores, bbox_preds, iou_preds

    def _format_preds_before_loss(self, cls_scores, bbox_preds, iou_preds,
                                  labels, label_weight, bboxes_target,
                                  anchor_list):
        """Format preds before computing loss"""
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        flatten_anchors = torch.cat([torch.cat(item, 0)
                                     for item in anchor_list])
        labels = torch.cat(labels, 0).view(-1)
        labels_weight = torch.cat(label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target, 0).view(
            -1, bboxes_target[0].size(-1))
        return (cls_scores, bbox_preds, iou_preds, flatten_anchors,
                labels, labels_weight, bboxes_target)

    def _preassign(self, gt_bboxes, gt_labels, gt_bboxes_ignore, img_metas,
                   anchor_list, valid_flag_list):
        """Pre-assign for all imgs in batch"""
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        preassign_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        return preassign_targets

    def _compute_loss(self, cls_scores, bbox_preds, iou_preds, flatten_anchors,
                      labels, labels_weight, bboxes_target, num_pos, img_metas):
        """Compute loss for classification, bbox, and iou"""
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

    def paa_reassign(self, pos_losses, label, label_weight, bbox_weight,
                     pos_inds, pos_gt_inds, anchors):
        """Perform re-assigment for an image in batch"""
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)

        # collect level mask for positive mask
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        # collect pos_inds and ignore_inds of gt objs
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                if len(pos_inds_gmm) == 1:
                    pos_inds_after_paa.append(pos_inds_gmm)
                continue
            # fit GMM
            pos_inds_temp, ignore_inds_temp = self._fit_gmm(pos_loss_gmm,
                                                            pos_inds_gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)

        # gather pos_inds and ignore_inds for all gt objs
        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)

        # re-assign
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, bbox_weight, num_pos

    def _fit_gmm(self, pos_loss_gmm, pos_inds_gmm):
        """GMM fits to pos_losses of a gt obj"""
        # normalize loss may stabilize the GMM fittig
        if self.norm_loss:
            min_val = pos_loss_gmm.min()
            max_val = pos_loss_gmm.max()
            pos_loss_gmm = (pos_loss_gmm - min_val) / (max_val - min_val + EPS)
        device = pos_inds_gmm.device
        pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
        pos_inds_gmm = pos_inds_gmm[sort_inds]
        pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
        min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
        means_init = np.array([min_loss, max_loss]).reshape(2, 1)
        weights_init = np.array([0.5, 0.5])
        precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
        if self.covariance_type == 'spherical':
            precisions_init = precisions_init.reshape(2)
        elif self.covariance_type == 'diag':
            precisions_init = precisions_init.reshape(2, 1)
        elif self.covariance_type == 'tied':
            precisions_init = np.array([[1.0]])
        if skm is None:
            raise ImportError('Please run "pip install sklearn" '
                              'to install sklearn first.')
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            covariance_type=self.covariance_type)
        try:
            gmm.fit(pos_loss_gmm)
        except:
            print("GMM fitting error")
        gmm_assignment = gmm.predict(pos_loss_gmm)
        scores = gmm.score_samples(pos_loss_gmm)
        gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
        scores = torch.from_numpy(scores).to(device)

        pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
            gmm_assignment, scores, pos_inds_gmm, pos_loss_gmm, gmm)
        return pos_inds_temp, ignore_inds_temp

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm,
                              pos_loss_gmm, gmm):
        """Input `pos_loss_gmm` and `gmm` for further processing"""
        return super().gmm_separation_scheme(gmm_assignment, scores,
                                             pos_inds_gmm)

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        fusion_cfg = self.test_cfg.get('fusion_cfg', None)
        fusion_method = fusion_cfg.type if fusion_cfg else 'simple'
        if fusion_method == 'simple':
            _det_bboxes, det_labels = self.aug_test_bboxes_simple(
                feats, img_metas, rescale)
        elif fusion_method == 'soft_vote':
            _det_bboxes, det_labels = self.aug_test_bboxes_vote(
                feats, img_metas, rescale)
        else:
            raise ValueError('Unknown TTA fusion method')
        bbox_results = bbox2result(_det_bboxes, det_labels, self.num_classes)
        return bbox_results

    def aug_test_bboxes_simple(self, feats, img_metas, rescale=False):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        if hasattr(self, '_get_bboxes'):
            gbs_sig = signature(self._get_bboxes)
        else:
            gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        return _det_bboxes, det_labels

    def merge_aug_vote_results(self, aug_bboxes, aug_labels, img_metas):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_labels is None:
            return bboxes
        else:
            labels = torch.cat(aug_labels, dim=0)
            return bboxes, labels

    def remove_boxes(self, boxes, min_scale, max_scale):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        in_range_idxs = torch.nonzero(
            (areas >= min_scale * min_scale) &
            (areas <= max_scale * max_scale),
            as_tuple=False).squeeze(1)
        return in_range_idxs

    def vote_bboxes(self, boxes, scores, vote_thresh=0.66):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        assert self.test_cfg.fusion_cfg.type == 'soft_vote'
        eps = 1e-6
        score_thr = self.test_cfg.score_thr

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy().reshape(-1, 1)
        det = np.concatenate((boxes, scores), axis=1)
        if det.shape[0] <= 1:
            return np.zeros((0, 5)), np.zeros((0, 1))
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = area[0] + area[:] - inter
            union = np.maximum(union, eps)
            o = inter / union
            o[0] = 1

            # get needed merge det and delete these det
            merge_index = np.where(o >= vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except ValueError:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= score_thr)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                max_score = np.max(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = max_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except ValueError:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]
        boxes = torch.from_numpy(dets[:, :4]).float().cuda()
        scores = torch.from_numpy(dets[:, 4]).float().cuda()
        return boxes, scores

    def aug_test_bboxes_vote(self, feats, img_metas, rescale=False):
        """https://github.com/shinya7y/UniverseNet/blob/c622acfc3ceca3fe9df86a8db2d04b73562a02ac/mmdet/models/dense_heads/dense_test_mixins.py"""
        scale_ranges = self.test_cfg.fusion_cfg.scale_ranges
        num_same_scale_tta = len(feats) // len(scale_ranges)
        aug_bboxes = []
        aug_labels = []
        for aug_idx, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            det_bboxes, det_labels = self.get_bboxes(*bbox_inputs)[0]
            min_scale, max_scale = scale_ranges[aug_idx // num_same_scale_tta]
            in_range_idxs = self.remove_boxes(det_bboxes, min_scale, max_scale)
            det_bboxes = det_bboxes[in_range_idxs, :]
            det_labels = det_labels[in_range_idxs]
            aug_bboxes.append(det_bboxes)
            aug_labels.append(det_labels)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_labels = self.merge_aug_vote_results(
            aug_bboxes, aug_labels, img_metas)

        det_bboxes = []
        det_labels = []
        for j in range(self.num_classes):
            inds = (merged_labels == j).nonzero(as_tuple=False).squeeze(1)

            scores_j = merged_bboxes[inds, 4]
            bboxes_j = merged_bboxes[inds, :4].view(-1, 4)
            bboxes_j, scores_j = self.vote_bboxes(bboxes_j, scores_j)

            if len(bboxes_j) > 0:
                det_bboxes.append(
                    torch.cat([bboxes_j, scores_j[:, None]], dim=1))
                det_labels.append(
                    torch.full((bboxes_j.shape[0], ),
                               j,
                               dtype=torch.int64,
                               device=scores_j.device))

        if len(det_bboxes) > 0:
            det_bboxes = torch.cat(det_bboxes, dim=0)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = merged_bboxes.new_zeros((0, 5))
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

        max_per_img = self.test_cfg.max_per_img
        if det_bboxes.shape[0] > max_per_img > 0:
            cls_scores = det_bboxes[:, 4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), det_bboxes.shape[0] - max_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        return _det_bboxes, det_labels


@HEADS.register_module()
class PAA_CoLAD_StdMean_Head(StandardizedPAAHead):
    """PAA Head for Co-learning Label Assignment Distillation (CoLAD)
    with Std/Mean criterion https://arxiv.org/abs/2108.10520"""
    """Co-Learning PAA Head with Std/Mean criterion"""

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
                bboxes_target, bboxes_weight, bbox_preds,
                pos_inds, pos_gt_index)

    def _label_reassign(self,
                        pos_losses_list,
                        t_pos_losses_list,
                        anchor_list,
                        labels,
                        labels_weight,
                        bboxes_target,
                        bboxes_weight,
                        pos_inds,
                        pos_gt_index):
        with torch.no_grad():
            (reassign_labels, reassign_label_weight, reassign_bbox_weights,
             num_pos, s_means, s_stds, t_means, t_stds, t_ratios) = multi_apply(
                self.paa_reassign,
                pos_losses_list,
                t_pos_losses_list,
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
        # dynamic statistics
        s_mean = torch.stack(s_means).mean()
        s_std = torch.stack(s_stds).mean()
        t_mean = torch.stack(t_means).mean()
        t_std = torch.stack(t_stds).mean()
        t_ratio = torch.stack(t_ratios).mean()
        dynamic_statistics = dict(s_mean=s_mean, s_std=s_std,
                                  t_mean=t_mean, t_std=t_std,
                                  t_ratio=t_ratio)
        # return
        return (labels, labels_weight, bboxes_target, bboxes_weight,
                pos_inds_flatten, pos_anchors, num_pos, dynamic_statistics)

    def paa_reassign(self, pos_losses, teacher_pos_losses,
                     label, label_weight, bbox_weight,
                     pos_inds, pos_gt_inds, anchors):
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        s_means, s_stds, t_means, t_stds, t_counts = [], [], [], [], []
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                s_pos_loss = pos_losses[level_gt_mask]
                t_pos_loss = teacher_pos_losses[level_gt_mask]
                # select teacher or student
                pos_loss = s_pos_loss
                if torch.any(level_gt_mask):
                    s_mean, t_mean = s_pos_loss.mean(), t_pos_loss.mean()
                    s_means.append(s_mean)
                    t_means.append(t_mean)
                    if len(s_pos_loss) > 1:
                        s_std, t_std = s_pos_loss.std(), t_pos_loss.std()
                        s_stds.append(s_std)
                        t_stds.append(t_std)
                        if t_std/t_mean >= s_std/s_mean:
                            pos_loss = t_pos_loss
                            t_counts.append(1)
                        else:
                            t_counts.append(0)
                    else:
                        if t_mean < s_mean:
                            pos_loss = t_pos_loss
                            t_counts.append(1)
                        else:
                            t_counts.append(0)
                value, topk_inds = pos_loss.topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type)
            try:
                gmm.fit(pos_loss_gmm)
            except:
                continue
            gmm_assignment = gmm.predict(pos_loss_gmm)
            scores = gmm.score_samples(pos_loss_gmm)
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)

            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm, pos_loss_gmm, gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)

        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)

        # teacher/student statistics
        s_mean, s_std = torch.stack(s_means).mean(), torch.stack(s_stds).mean()
        t_mean, t_std = torch.stack(t_means).mean(), torch.stack(t_stds).mean()
        t_ratio = s_mean.new_tensor(t_counts).mean()

        # return
        return (label, label_weight, bbox_weight, num_pos,
                s_mean, s_std, t_mean, t_std, t_ratio)

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
