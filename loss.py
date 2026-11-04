import torch
import torch.nn.functional as F
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box1, box2):

        # Convert center coordinates to corner coordinates
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        # Intersection coordinates
        inter_x1 = torch.maximum(b1_x1, b2_x1)
        inter_y1 = torch.maximum(b1_y1, b2_y1)
        inter_x2 = torch.minimum(b1_x2, b2_x2)
        inter_y2 = torch.minimum(b1_y2, b2_y2)

        # Intersection area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

        intersection = inter_w * inter_h

        # Individual areas
        area1 = box1[..., 2] * box1[..., 3]
        area2 = box2[..., 2] * box2[..., 3]

        # Union
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)


    def forward(self, pred, target):
        N = pred.size(0)

        # -------------------------------------------------------------
        # 1. Indicator Mask: which cells contain an object?
        # -------------------------------------------------------------
        # target[..., 4:5] has shape (N, S, S, 1): 1 if object exists, 0 otherwise
        exists_box = target[..., 4:5]

        # -------------------------------------------------------------
        # 2. Responsible Box Selection (Higher IoU with GT)
        # -------------------------------------------------------------
        # Convert w, h from image units [0, 1] to cell units [0, S]
        # so they share the same scale as cell offsets (x, y)
        gt_box = target[..., 0:4].clone()
        gt_box[..., 2:4] *= self.S

        p_box1 = pred[..., 0:4].clone()
        p_box1[..., 2:4] *= self.S

        p_box2 = pred[..., 5:9].clone()
        p_box2[..., 2:4] *= self.S

        iou1 = self.compute_iou(p_box1, gt_box).unsqueeze(-1)  # (N, S, S, 1)
        iou2 = self.compute_iou(p_box2, gt_box).unsqueeze(-1)  # (N, S, S, 1)

        # best_box = 1 if Box 2 has higher IoU, else 0 (Box 1 is better)
        # We detach so we don't backprop through the decision of who won
        best_box = (iou2 > iou1).float().detach()

        # Gather the coordinates & confidence of whichever box won:
        # If best_box == 0: selects Box 1
        # If best_box == 1: selects Box 2
        pred_responsible_box = (1 - best_box) * pred[..., 0:4] + best_box * pred[..., 5:9]
        pred_responsible_conf = (1 - best_box) * pred[..., 4:5] + best_box * pred[..., 9:10]

        # -------------------------------------------------------------
        # 3. Loss Term 1 & 2: Coordinate Loss (x, y) and (sqrt(w), sqrt(h))
        # -------------------------------------------------------------
        # (x, y) center loss
        loss_xy = self.lambda_coord * torch.sum(
            exists_box * (pred_responsible_box[..., 0:2] - target[..., 0:2]) ** 2
        )

        # (w, h) size loss: square root difference
        # clamp to 1e-6 to prevent sqrt(0) which causes NaN gradients!
        pred_wh = torch.sqrt(torch.clamp(pred_responsible_box[..., 2: 4],  min=1e-6))
        target_wh = torch.sqrt(target[..., 2:4])
        loss_wh = self.lambda_coord * torch.sum(
            exists_box * (pred_wh - target_wh) ** 2
        )

        # -------------------------------------------------------------
        # 4. Loss Term 3: Object Confidence Loss
        # -------------------------------------------------------------
        # Target confidence is 1.0 (or target IoU) for the responsible box
        loss_obj = torch.sum(
            exists_box * (pred_responsible_conf - 1)**2
        )

        # -------------------------------------------------------------
        # 5. Loss Term 4: No-Object Confidence Loss
        # -------------------------------------------------------------
        # Penalizes:
        # - BOTH boxes in empty cells (exists_box == 0)
        # - The LOSING box in cells with an object (exists_box == 1)
        noobj_mask_box1 = (1 - exists_box) + exists_box * best_box       # 1 if box1 should be 0
        noobj_mask_box2 = (1 - exists_box) + exists_box * (1 - best_box) # 1 if box2 should be 0

        loss_noobj = self.lambda_noobj * (
            torch.sum(noobj_mask_box1 * (pred[..., 4:5] - 0.0) ** 2) +
            torch.sum(noobj_mask_box2 * (pred[..., 9:10] - 0.0) ** 2)
        )

        # -------------------------------------------------------------
        # 6. Loss Term 5: Classification Loss
        # -------------------------------------------------------------
        loss_class = torch.sum(
            exists_box * (pred[..., 10:30] - target[..., 10:30]) ** 2
        )

        # Total Loss
        total_loss = (loss_xy + loss_wh + loss_obj + loss_noobj + loss_class) / N
        return total_loss
    
