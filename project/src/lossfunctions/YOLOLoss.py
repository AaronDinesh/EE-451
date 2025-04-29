import torch.nn.functional as F

def yolo_loss(pred, target):
    # Reshape to match target
    pred = pred.permute(0, 1, 3, 4, 2)  # [B, A, S, S, 5+C]
    
    # Components
    pred_box = pred[..., 0:4]
    pred_obj = pred[..., 4]
    pred_cls = pred[..., 5:]

    true_box = target[..., 0:4]
    true_obj = target[..., 4]
    true_cls = target[..., 5:]

    # Coordinate loss (only where object exists)
    coord_loss = F.mse_loss(pred_box[true_obj == 1], true_box[true_obj == 1])

    # Objectness loss
    obj_loss = F.binary_cross_entropy_with_logits(pred_obj, true_obj)

    # Classification loss
    cls_loss = F.binary_cross_entropy_with_logits(pred_cls[true_obj == 1], true_cls[true_obj == 1])

    return coord_loss + obj_loss + cls_loss
