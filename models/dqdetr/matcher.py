# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .utils import linear_epoch_ramp


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
        cost_pmap: float = 0.0,
        cost_pmap_eps: float = 1e-6,
        cost_pmap_warmup_epochs: int = 0,
        cost_pmap_ramp_epochs: int = 0,
        cost_pmap_q_mode: str = "localmax",
        cost_pmap_localmax_r: int = 2,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        self.cost_pmap = float(cost_pmap)
        self.cost_pmap_eps = float(cost_pmap_eps)
        self.cost_pmap_warmup_epochs = int(cost_pmap_warmup_epochs)
        self.cost_pmap_ramp_epochs = int(cost_pmap_ramp_epochs)
        self.cost_pmap_q_mode = str(cost_pmap_q_mode)
        self.cost_pmap_localmax_r = int(cost_pmap_localmax_r)
        self._cur_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._cur_epoch = int(epoch)

    def get_cost_pmap_scale(self) -> float:
        return linear_epoch_ramp(
            self._cur_epoch,
            self.cost_pmap_warmup_epochs,
            self.cost_pmap_ramp_epochs,
        )

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_pmap_eff = self.cost_pmap * self.get_cost_pmap_scale()
        if cost_pmap_eff > 0.0 and isinstance(outputs, dict):
            den = outputs.get("dense_heatmap_lvl0", None)
            if (
                isinstance(den, torch.Tensor)
                and den.dim() == 4
                and den.size(0) == bs
                and den.size(1) == 1
            ):
                den_f = den.detach().to(dtype=torch.float32)
                mass = den_f.sum(dim=(1, 2, 3)).clamp_min(1e-12).view(-1, 1, 1, 1)
                q_map = den_f / mass

                if self.cost_pmap_q_mode.lower() == "localmax":
                    r = max(0, int(self.cost_pmap_localmax_r))
                    if r > 0:
                        ksz = 2 * r + 1
                        q_map = F.max_pool2d(q_map, kernel_size=ksz, stride=1, padding=r)

                H0 = int(q_map.shape[2])
                W0 = int(q_map.shape[3])
                cx = out_bbox[:, 0].clamp(0.0, 1.0)
                cy = out_bbox[:, 1].clamp(0.0, 1.0)
                px = torch.floor(cx * float(W0)).to(dtype=torch.long).clamp(0, W0 - 1)
                py = torch.floor(cy * float(H0)).to(dtype=torch.long).clamp(0, H0 - 1)
                b_idx = (
                    torch.arange(bs, device=out_bbox.device)
                    .view(bs, 1)
                    .repeat(1, num_queries)
                    .view(-1)
                )
                q = q_map[b_idx, 0, py, px].clamp_min(0.0)
                cp = -torch.log(q + float(self.cost_pmap_eps))  # [B*Q]
                C = C + (float(cost_pmap_eff) * cp).unsqueeze(1)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
        cost_pmap: float = 0.0,
        cost_pmap_eps: float = 1e-6,
        cost_pmap_warmup_epochs: int = 0,
        cost_pmap_ramp_epochs: int = 0,
        cost_pmap_q_mode: str = "localmax",
        cost_pmap_localmax_r: int = 2,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        self.cost_pmap = float(cost_pmap)
        self.cost_pmap_eps = float(cost_pmap_eps)
        self.cost_pmap_warmup_epochs = int(cost_pmap_warmup_epochs)
        self.cost_pmap_ramp_epochs = int(cost_pmap_ramp_epochs)
        self.cost_pmap_q_mode = str(cost_pmap_q_mode)
        self.cost_pmap_localmax_r = int(cost_pmap_localmax_r)
        self._cur_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._cur_epoch = int(epoch)

    def get_cost_pmap_scale(self) -> float:
        return linear_epoch_ramp(
            self._cur_epoch,
            self.cost_pmap_warmup_epochs,
            self.cost_pmap_ramp_epochs,
        )

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_pmap_eff = self.cost_pmap * self.get_cost_pmap_scale()
        if cost_pmap_eff > 0.0 and isinstance(outputs, dict):
            den = outputs.get("dense_heatmap_lvl0", None)
            if (
                isinstance(den, torch.Tensor)
                and den.dim() == 4
                and den.size(0) == bs
                and den.size(1) == 1
            ):
                den_f = den.detach().to(dtype=torch.float32)
                mass = den_f.sum(dim=(1, 2, 3)).clamp_min(1e-12).view(-1, 1, 1, 1)
                q_map = den_f / mass

                if self.cost_pmap_q_mode.lower() == "localmax":
                    r = max(0, int(self.cost_pmap_localmax_r))
                    if r > 0:
                        ksz = 2 * r + 1
                        q_map = F.max_pool2d(q_map, kernel_size=ksz, stride=1, padding=r)

                H0 = int(q_map.shape[2])
                W0 = int(q_map.shape[3])
                cx = out_bbox[:, 0].clamp(0.0, 1.0)
                cy = out_bbox[:, 1].clamp(0.0, 1.0)
                px = torch.floor(cx * float(W0)).to(dtype=torch.long).clamp(0, W0 - 1)
                py = torch.floor(cy * float(H0)).to(dtype=torch.long).clamp(0, H0 - 1)
                b_idx = (
                    torch.arange(bs, device=out_bbox.device)
                    .view(bs, 1)
                    .repeat(1, num_queries)
                    .view(-1)
                )
                q = q_map[b_idx, 0, py, px].clamp_min(0.0)
                cp = -torch.log(q + float(self.cost_pmap_eps))  # [B*Q]
                C = C + (float(cost_pmap_eff) * cp).unsqueeze(1)
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    cost_kwargs = dict(
        cost_pmap=getattr(args, "set_cost_pmap", 0.0),
        cost_pmap_eps=getattr(args, "cost_pmap_eps", 1e-6),
        cost_pmap_warmup_epochs=getattr(args, "cost_pmap_warmup_epochs", 0),
        cost_pmap_ramp_epochs=getattr(args, "cost_pmap_ramp_epochs", 0),
        cost_pmap_q_mode=getattr(args, "cost_pmap_q_mode", "localmax"),
        cost_pmap_localmax_r=getattr(args, "cost_pmap_localmax_r", 2),
    )
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            **cost_kwargs,
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            **cost_kwargs,
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))
