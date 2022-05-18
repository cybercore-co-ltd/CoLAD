import torch

from mmcv.runner import force_fp32

from mmdet.models.builder import build_loss
from mmdet.models import HEADS, PAAHead
from mmdet.core import bbox_overlaps
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12


@HEADS.register_module()
class PAA_KD_Head(PAAHead):

    def __init__(self,
                 kd_cls_loss=None,
                 kd_bbox_loss=None,
                 kd_centerness_loss=None,
                 min_iou=0.1,
                 use_pred_bbox=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # buid KD losses
        self.kd_cls_loss = build_loss(kd_cls_loss)
        self.kd_bbox_loss = build_loss(kd_bbox_loss)
        self.kd_centerness_loss = build_loss(kd_centerness_loss)
        self.min_iou = min_iou
        self.use_pred_bbox = use_pred_bbox

    @property
    def with_kd_cls(self):
        """bool: whether the head has kd_cls"""
        return hasattr(self, 'kd_cls_loss') and self.kd_cls_loss is not None

    @property
    def with_kd_bbox(self):
        """bool: whether the head has kd_bbox"""
        return hasattr(self, 'kd_bbox_loss') and self.kd_bbox_loss is not None

    @property
    def with_kd_iou(self):
        """bool: whether the head has kd_bbox"""
        return hasattr(self, 'kd_centerness_loss') \
            and self.kd_centerness_loss is not None

    def forward_train(self,
                      x,
                      teacher_outs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      proposal_cfg=None):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            teacher_outs (list[Tensor): Feature outputs of teacher network
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + teacher_outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + teacher_outs + (gt_bboxes, gt_labels,
                                                 img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_target_with_iou(self, anchor_list, valid_flag_list,
                            bbox_preds, teacher_bbox_preds,
                            iou_preds, teacher_iou_preds,
                            gt_bboxes, min_iou, img_metas):
        """Get target.

        Args:
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            teacher_bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            teacher_iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            min_iou (float): min IOU of predicted boxes w.r.t groundtruth 
                one so that the candidates are positive
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            tuple(Tensor): candidates for computing loss.
        """
        num_imgs = len(img_metas)
        num_levels = len(bbox_preds)

        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # convert pred per level to pred per image
        bbox_pred_list = []
        teacher_bbox_pred_list = []
        st_iou_pred_list = []
        teacher_iou_pred_list = []
        for img_id in range(num_imgs):
            # bbox
            bbox_pred_list.append([bbox_preds[i][img_id].permute(
                1, 2, 0).reshape(-1, 4) for i in range(num_levels)])
            teacher_bbox_pred_list.append([teacher_bbox_preds[i][img_id].detach(
            ).permute(1, 2, 0).reshape(-1, 4) for i in range(num_levels)])
            bbox_pred_list[img_id] = torch.cat(bbox_pred_list[img_id])
            teacher_bbox_pred_list[img_id] = torch.cat(
                teacher_bbox_pred_list[img_id])

            # iou
            st_iou_pred_list.append(
                [iou_preds[i][img_id].reshape(-1, 1) for i in range(num_levels)])
            teacher_iou_pred_list.append([teacher_iou_preds[i][img_id].detach(
            ).reshape(-1, 1) for i in range(num_levels)])
            st_iou_pred_list[img_id] = torch.cat(st_iou_pred_list[img_id])
            teacher_iou_pred_list[img_id] = torch.cat(
                teacher_iou_pred_list[img_id])

        st_bbox_preds_decode = []
        te_bbox_preds_decode = []
        st_iou_preds = []
        te_iou_preds = []
        loss_weight_list = []

        for img_id in range(num_imgs):
            img_shape = img_metas[img_id]['img_shape']

            # take valid bboxes only
            valid_ids = concat_valid_flag_list[img_id]
            valid_anchors = concat_anchor_list[img_id][valid_ids]
            valid_st_preds = bbox_pred_list[img_id][valid_ids]
            valid_te_preds = teacher_bbox_pred_list[img_id][valid_ids]
            img_gt_bboxes = gt_bboxes[img_id]

            # take valid iou pre only
            valid_st_iou_preds = st_iou_pred_list[img_id][valid_ids]
            valid_te_iou_preds = teacher_iou_pred_list[img_id][valid_ids]

            # decode bboxes
            valid_te_preds_decode = self.bbox_coder.decode(valid_anchors,
                                                           valid_te_preds,
                                                           img_shape)
            valid_st_preds_decode = self.bbox_coder.decode(valid_anchors,
                                                           valid_st_preds,
                                                           img_shape)

            # Calculate ious
            if self.use_pred_bbox:
                ious = bbox_overlaps(valid_te_preds_decode, img_gt_bboxes)
            else:
                ious = bbox_overlaps(valid_anchors, img_gt_bboxes)

            # take box_preds which have iou with any gt > min_iou
            ious_valid = (ious > min_iou).float().sum(dim=1) > 0
            st_bbox_preds_decode.append(valid_st_preds_decode[ious_valid])
            te_bbox_preds_decode.append(valid_te_preds_decode[ious_valid])

            # take iou_preds which have iou with any gt > min_iou
            st_iou_preds.append(valid_st_iou_preds[ious_valid])
            te_iou_preds.append(valid_te_iou_preds[ious_valid])
            try:
                loss_weight_list.append(
                    ious[ious_valid].float().max(1)[0].reshape(-1, 1))
            except:
                pass

        # concatenate tensor
        st_bbox_preds_decode = torch.cat(st_bbox_preds_decode)
        te_bbox_preds_decode = torch.cat(te_bbox_preds_decode)
        st_iou_preds = torch.cat(st_iou_preds).reshape(-1)
        te_iou_preds = torch.cat(te_iou_preds).reshape(-1)
        loss_weight = torch.cat(loss_weight_list).reshape(-1)

        return (st_bbox_preds_decode, te_bbox_preds_decode,
                st_iou_preds, te_iou_preds, loss_weight)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds',
                          'teacher_cls_scores', 'teacher_bbox_preds',
                          'teacher_iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             teacher_cls_scores,
             teacher_bbox_preds,
             teacher_iou_preds,
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
            teacher_cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            teacher_bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            teacher_iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when are computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        min_iou = self.min_iou
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        # Get target
        (st_bbox_preds_decode, te_bbox_preds_decode,
         st_iou_preds, te_iou_preds, reg_oss_weight) = \
            self.get_target_with_iou(
                anchor_list=anchor_list,
                valid_flag_list=valid_flag_list,
                bbox_preds=bbox_preds,
                teacher_bbox_preds=teacher_bbox_preds,
                iou_preds=iou_preds,
                teacher_iou_preds=teacher_iou_preds,
                gt_bboxes=gt_bboxes,
                min_iou=min_iou,
                img_metas=img_metas)
        average_factor = reg_oss_weight.sum()

        # convert all list of tensor to a flatten tensor
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, 1) for item in cls_scores]
        cls_scores = torch.cat(cls_scores, 0).reshape(-1)

        teacher_cls_scores = levels_to_images(teacher_cls_scores)
        teacher_cls_scores = [item.reshape(-1, 1)
                              for item in teacher_cls_scores]
        teacher_cls_scores = torch.cat(teacher_cls_scores, 0).reshape(-1)
        losses = dict()

        # distill cls loss
        losses_cls = self.kd_cls_loss(
            cls_scores,
            teacher_cls_scores.sigmoid(),
            avg_factor=average_factor)

        # distillation regression loss
        if len(st_bbox_preds_decode):
            losses_centerness = self.kd_centerness_loss(
                st_iou_preds,
                te_iou_preds.sigmoid(),
                avg_factor=average_factor)
            losses_bbox = self.kd_bbox_loss(
                st_bbox_preds_decode,
                te_bbox_preds_decode,
                reg_oss_weight.clamp(min=EPS),
                avg_factor=average_factor)
        else:
            losses_centerness = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        # update loss
        losses.update(loss_kd_bbox=losses_bbox,
                      loss_kd_centerness=losses_centerness,
                      loss_kd_cls=losses_cls)
        return losses
