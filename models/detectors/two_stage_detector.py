import torch
import torch.nn as nn

from .base_detector import BaseDetector


class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(
            self,
            backbone: nn.Module,
            neck: nn.Module = None,
            rpn_head: nn.Module = None,
            roi_head: nn.Module = None,
    ):
        super(TwoStageDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.rpn_head:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward(
            self,
            imgs,
            img_metas=None,
            gt_bboxes=None,
            gt_labels=None,
            gt_masks=None,
            proposals=None,
    ):
        """
        Args:
            imgs (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(imgs)

        losses = dict()

        # RPN forward and loss
        if self.rpn_head:
            if proposals is not None:
                proposal_list = proposals
            else:
                proposal_list, rpn_losses = self.rpn_head(x, img_metas, gt_bboxes)
                losses.update(rpn_losses)
        else:
            assert proposals is not None, "proposals must be provided when rpn_head=None"
            proposal_list = proposals

        # roi_head forward and loss
        cls_score, bbox_pred, roi_losses = self.roi_head(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_masks
        )
        losses.update(roi_losses)

        return proposal_list, cls_score, bbox_pred, losses
