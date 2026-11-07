import xml.etree.ElementTree as ET
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def parse_xml(path):
    root = ET.parse(path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        b = obj.find("bndbox")
        xmin = int(b.find("xmin").text)
        ymin = int(b.find("ymin").text)
        xmax = int(b.find("xmax").text)
        ymax = int(b.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax, difficult))
    return w, h, boxes


def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    """
    Calculates IoU between two sets of boxes.
    Supports either [x1, y1, x2, y2] ("corners") or [x, y, w, h] ("midpoint").
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def cellboxes_to_boxes(out, S=7, device="cpu"):
    """
    Converts model output tensor (N, S, S, 30) into a list of bounding boxes
    per image: [class_pred, score, x1, y1, x2, y2]
    """
    batch_size = out.shape[0]
    out = out.to(device)
    
    # 1. Best predicted class and its probability for every cell
    class_probs = out[..., 10:30]
    best_class = torch.argmax(class_probs, dim=-1).unsqueeze(-1)  # (N, S, S, 1)
    best_prob = torch.max(class_probs, dim=-1).values.unsqueeze(-1) # (N, S, S, 1)

    # 2. Extract Box 1 and Box 2 coordinates & confidences
    # For Box 1:
    b1_x = out[..., 0:1]
    b1_y = out[..., 1:2]
    b1_w = out[..., 2:3]
    b1_h = out[..., 3:4]
    b1_conf = out[..., 4:5] * best_prob  # Final score = conf * class_prob

    # For Box 2:
    b2_x = out[..., 5:6]
    b2_y = out[..., 6:7]
    b2_w = out[..., 7:8]
    b2_h = out[..., 8:9]
    b2_conf = out[..., 9:10] * best_prob

    # 3. Create grid offsets (col, row)
    # col: (S, S), row: (S, S)
    cell_indices = torch.arange(S, device=out.device).repeat(S, 1)
    col = cell_indices.unsqueeze(0).unsqueeze(-1)  # (1, S, S, 1)
    row = cell_indices.t().unsqueeze(0).unsqueeze(-1)

    # Convert to image-scale [0, 1] coordinates
    b1_x_global = (col + b1_x) / S
    b1_y_global = (row + b1_y) / S
    b2_x_global = (col + b2_x) / S
    b2_y_global = (row + b2_y) / S

    # Convert center (x, y, w, h) to corner (x1, y1, x2, y2)
    b1_x1, b1_y1 = b1_x_global - b1_w / 2, b1_y_global - b1_h / 2
    b1_x2, b1_y2 = b1_x_global + b1_w / 2, b1_y_global + b1_h / 2

    b2_x1, b2_y1 = b2_x_global - b2_w / 2, b2_y_global - b2_h / 2
    b2_x2, b2_y2 = b2_x_global + b2_w / 2, b2_y_global + b2_h / 2

    # Pack into (N, S, S, 6): [class, score, x1, y1, x2, y2]
    box1 = torch.cat([best_class.float(), b1_conf, b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
    box2 = torch.cat([best_class.float(), b2_conf, b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)

    # Flatten across S x S grid: each image gets 2 * S * S = 98 candidate boxes
    box1 = box1.reshape(batch_size, S * S, 6)
    box2 = box2.reshape(batch_size, S * S, 6)
    all_boxes = torch.cat([box1, box2], dim=1)  # (N, 98, 6)

    # Return list of list of boxes per image
    return all_boxes.tolist()


def non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4):
    """
    Filters candidate boxes for a SINGLE image.
    bboxes: list of [class_pred, score, x1, y1, x2, y2]
    """
    # 1. Filter out low-confidence boxes
    bboxes = [box for box in bboxes if box[1] > threshold]

    # 2. Sort boxes by confidence in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # Keep chosen_box, and discard any box of the SAME class with IoU > iou_threshold
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format="corners"
            ).item() < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_boxes(image, boxes, class_names):
    """
    Plots an image with its predicted bounding boxes.
    image: (3, H, W) normalized tensor
    boxes: list of [class_pred, score, x1, y1, x2, y2]
    """
    # Un-normalize image back to [0, 1] for display
    inv_normalize = image.clone()
    inv_normalize[0] = inv_normalize[0] * 0.229 + 0.485
    inv_normalize[1] = inv_normalize[1] * 0.224 + 0.456
    inv_normalize[2] = inv_normalize[2] * 0.225 + 0.406
    np_img = inv_normalize.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(np_img)
    H, W, _ = np_img.shape

    for box in boxes:
        cls_idx, score, x1, y1, x2, y2 = box
        # Scale normalized [0, 1] coords to pixel coordinates
        box_w = (x2 - x1) * W
        box_h = (y2 - y1) * H
        rect = patches.Rectangle(
            (x1 * W, y1 * H),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1 * W,
            y1 * H - 5,
            f"{class_names[int(cls_idx)]}: {score:.2f}",
            color="black",
            fontsize=10,
            backgroundcolor="lime",
        )

    plt.axis("off")
    plt.show()
    

if __name__ == "__main__":
    dummy_pred = torch.rand(1, 7, 7, 30)
    decoded = cellboxes_to_boxes(dummy_pred)
    print(f"Candidate boxes for image 0: {len(decoded[0])}")  # Should be 98
    
    nms_boxes = non_max_suppression(decoded[0], threshold=0.3)
    print(f"Boxes after NMS: {len(nms_boxes)}")