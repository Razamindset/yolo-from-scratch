import torch

def iou(box1, box2):

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

def reasonable_box(pred, target):

    box1 = pred[..., 0:4].clone()
    box2 = pred[..., 5:9].clone()

    gt = target[..., 0:4].clone()

    # Convert width and height to cell units
    box1[..., 2:4] *= 7
    box2[..., 2:4] *= 7
    gt[..., 2:4] *= 7

    # Calculate IoU for both predicted boxes
    iou1 = iou(box1, gt)
    iou2 = iou(box2, gt)

    # Stack IoUs and select the winner
    best_iou, best = torch.stack([iou1, iou2], dim=-1).max(dim=-1)

    return best, best_iou