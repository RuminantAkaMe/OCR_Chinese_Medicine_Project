import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import json

TARGET_CLASS_ID = None  # Set this if your model has multiple classes

def extract_boxes_from_json(gt_json_path):
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for obj in data.get("objects", []):
        points = obj.get("points", {}).get("exterior", None)
        if points and len(points) == 2:
            (x1, y1), (x2, y2) = points
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def calculate_average_precision(recalls, precisions):
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def evaluate_prediction(pred_image_path, gt_json_path, model_path,
                        iou_threshold=0.3, conf_threshold=0.4, pad_pixels=4):
    
    os.makedirs("Hardik_Final", exist_ok=True)
    model = YOLO(model_path)

    img = cv2.imread(pred_image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Image not found: {pred_image_path}")
    img_h, img_w = img.shape[:2]

    results = model.predict(source=img, save=False, conf=conf_threshold)
    boxes_all = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    classes_all = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
    scores_all = results[0].boxes.conf.cpu().numpy()

    if TARGET_CLASS_ID is not None:
        filtered = [(b, s) for b, c, s in zip(boxes_all, classes_all, scores_all) if c == TARGET_CLASS_ID]
        boxes_all, scores_all = zip(*filtered) if filtered else ([], [])
    else:
        boxes_all = boxes_all
        scores_all = scores_all

    # Apply NMS
    if boxes_all:
        boxes_tensor = torch.tensor(boxes_all, dtype=torch.float32)
        scores_tensor = torch.tensor(scores_all, dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        boxes_all = [boxes_all[i] for i in keep]
        scores_all = [scores_all[i] for i in keep]

    # Pad boxes with bounds check
    pred_boxes_padded = []
    for box in boxes_all:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad_pixels)
        y1 = max(0, y1 - pad_pixels)
        x2 = min(img_w, x2 + pad_pixels)
        y2 = min(img_h, y2 + pad_pixels)
        pred_boxes_padded.append([x1, y1, x2, y2])

    gt_boxes = extract_boxes_from_json(gt_json_path)
    print(f"[DEBUG] GT boxes: {len(gt_boxes)}, Predicted boxes: {len(pred_boxes_padded)}")

    if scores_all:
        sorted_indices = np.argsort(-np.array(scores_all))
        pred_boxes_sorted = [pred_boxes_padded[i] for i in sorted_indices]
        scores_sorted = [scores_all[i] for i in sorted_indices]
    else:
        pred_boxes_sorted = []
        scores_sorted = []

    matched_gt = set()
    tp_list, fp_list, ious = [], [], []

    for pb in pred_boxes_sorted:
        match_found = False
        for j, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou >= iou_threshold and j not in matched_gt:
                matched_gt.add(j)
                ious.append(iou)
                match_found = True
                break
        tp_list.append(1 if match_found else 0)
        fp_list.append(0 if match_found else 1)

    FN = len(gt_boxes) - len(matched_gt)
    TP = sum(tp_list)
    FP = sum(fp_list)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(ious) if ious else 0.0
    accuracy = TP / (TP + FP + FN + 1e-6)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    precisions_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls_curve = tp_cumsum / (len(gt_boxes) + 1e-6)
    ap = calculate_average_precision(recalls_curve, precisions_curve)

    # Final Results
    print("\n[INFO] Evaluation Metrics:")
    print(f"True Positives (TP)      : {TP}")
    print(f"False Positives (FP)     : {FP}")
    print(f"False Negatives (FN)     : {FN}")
    print(f"Precision                : {precision:.4f}")
    print(f"Recall                   : {recall:.4f}")
    print(f"F1 Score                 : {f1_score:.4f}")
    print(f"Mean IoU                 : {mean_iou:.4f}")
    print(f"Accuracy (simplified)    : {accuracy:.4f}")
    print(f"Average Precision (AP)   : {ap:.4f}\n")

    # Visualize predictions
    vis_img = img.copy()
    for i, pb in enumerate(pred_boxes_sorted):
        color = (0, 255, 0) if tp_list[i] == 1 else (0, 0, 255)
        x1, y1, x2, y2 = pb
        conf = scores_sorted[i]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"{conf:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for gb in gt_boxes:
        x1, y1, x2, y2 = gb
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite("Hardik_Final/result_debug.jpg", vis_img)

if __name__ == "__main__":
    evaluate_prediction(
        pred_image_path='LabeledImage/8_org.jpeg',
        gt_json_path='LabeledImage/8_org.jpeg.json',
        model_path='yolov10n.pt',
        iou_threshold=0.1,     # A realistic match threshold
        conf_threshold=0.5,    # Balanced confidence
        pad_pixels=4
    )
