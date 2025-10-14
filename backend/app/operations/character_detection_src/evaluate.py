import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import json
import matplotlib.pyplot as plt

# If you're detecting multiple classes, set this to the class ID you want to evaluate
# Leave as None if you want to evaluate all classes together
TARGET_CLASS_ID = None

def extract_boxes_from_json(gt_json_path):
    # Load the ground truth annotations from JSON file
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    boxes = []
    # Go through each annotated object in the JSON
    for obj in data.get("objects", []):
        # Get the bounding box coordinates (expecting two corner points)
        points = obj.get("points", {}).get("exterior", None)
        if points and len(points) == 2:
            # Convert the two corner points to x1, y1, x2, y2 format
            (x1, y1), (x2, y2) = points
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    return boxes

def compute_iou(boxA, boxB):
    # Calculate Intersection over Union between two boxes
    # This tells us how much two boxes overlap
    
    # Find the coordinates of the intersection rectangle
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    
    # Calculate intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Calculate area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # IoU = intersection / union
    # Adding 1e-6 to avoid division by zero
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def calculate_average_precision(recalls, precisions):
    # Calculate AP using the 11-point interpolation method
    # This is the standard way to compute mAP in object detection
    
    # Add boundary values to make the calculation easier
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    
    # For each recall level, use the maximum precision to the right
    # This smooths out the precision-recall curve
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate area under the precision-recall curve
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def plot_precision_recall_curve(precisions, recalls, ap, output_path):
    """
    Plot and save the precision-recall curve
    This helps visualize the trade-off between precision and recall
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the actual PR curve
    plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
    
    # Add grid for easier reading
    plt.grid(True, alpha=0.3)
    
    # Labels and title
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for Chinese Character Detection', fontsize=14)
    
    # Set axis limits
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    
    # Add legend
    plt.legend(loc='best')
    
    # Add a diagonal reference line
    plt.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.3, label='Random Baseline')
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Precision-Recall curve saved to {output_path}")

def analyze_confidence_thresholds(pred_image_path, gt_json_path, model_path, 
                                 iou_threshold=0.3, pad_pixels=4):
    """
    Analyze performance at different confidence thresholds
    This helps find the best threshold for your specific needs
    """
    model = YOLO(model_path)
    
    # Load image to get dimensions
    img = cv2.imread(pred_image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Image not found: {pred_image_path}")
    
    img_h, img_w = img.shape[:2]
    
    # Get predictions with very low confidence to catch everything
    # Always use file path to avoid numpy array issues
    results = model(pred_image_path, conf=0.01, verbose=False)
    
    boxes_all = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    scores_all = results[0].boxes.conf.cpu().numpy()
    
    # Load ground truth
    gt_boxes = extract_boxes_from_json(gt_json_path)
    
    # Test different confidence thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    threshold_results = []
    
    print("\n[INFO] Analyzing different confidence thresholds...")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Detections':<12}")
    print("-" * 70)
    
    for threshold in thresholds:
        # Filter predictions by this threshold
        filtered_indices = [i for i, s in enumerate(scores_all) if s >= threshold]
        filtered_boxes = [boxes_all[i] for i in filtered_indices]
        
        # Add padding to boxes
        pred_boxes_padded = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - pad_pixels)
            y1 = max(0, y1 - pad_pixels)
            x2 = min(img_w, x2 + pad_pixels)
            y2 = min(img_h, y2 + pad_pixels)
            pred_boxes_padded.append([x1, y1, x2, y2])
        
        # Calculate metrics for this threshold
        matched_gt = set()
        tp = 0
        
        for pb in pred_boxes_padded:
            for j, gb in enumerate(gt_boxes):
                if j not in matched_gt:
                    iou = compute_iou(pb, gb)
                    if iou >= iou_threshold:
                        matched_gt.add(j)
                        tp += 1
                        break
        
        fp = len(pred_boxes_padded) - tp
        fn = len(gt_boxes) - len(matched_gt)
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        threshold_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_detections': len(pred_boxes_padded)
        })
        
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {len(pred_boxes_padded):<12}")
    
    print("-" * 70)
    
    # Find best thresholds for different objectives
    best_f1 = max(threshold_results, key=lambda x: x['f1'])
    best_recall = max(threshold_results, key=lambda x: x['recall'])
    
    # Find threshold for high recall (e.g., 90%) with best precision
    high_recall_threshold = None
    for result in threshold_results:
        if result['recall'] >= 0.9:
            high_recall_threshold = result
            break
    
    print("\n[INFO] Recommended thresholds for different objectives:")
    print(f"Best F1 Score: threshold={best_f1['threshold']:.2f}, F1={best_f1['f1']:.4f}, "
          f"Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}")
    print(f"Maximum Recall: threshold={best_recall['threshold']:.2f}, "
          f"Precision={best_recall['precision']:.4f}, Recall={best_recall['recall']:.4f}")
    
    if high_recall_threshold:
        print(f"High Recall (≥90%): threshold={high_recall_threshold['threshold']:.2f}, "
              f"Precision={high_recall_threshold['precision']:.4f}, Recall={high_recall_threshold['recall']:.4f}")
    
    # Special note for Chinese OCR
    print("\n[INFO] For Chinese Character OCR:")
    print("Consider using a lower confidence threshold (0.2-0.3) to maximize recall")
    print("This ensures most characters are detected, even if some false positives occur")
    
    return threshold_results

def evaluate_prediction(pred_image_path, gt_json_path, model_path,
                        iou_threshold=0.3, conf_threshold=0.4, pad_pixels=4,
                        plot_pr_curve=True, analyze_thresholds=False):
    
    # Create output directory for visualization results
    os.makedirs("Hardik_Final", exist_ok=True)
    
    # Load the YOLO model
    model = YOLO(model_path)

    # Read the image we're going to evaluate
    img = cv2.imread(pred_image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Image not found: {pred_image_path}")
    
    # Get image dimensions - we'll need these for bounds checking later
    img_h, img_w = img.shape[:2]

    # Run detection on the image using file path to avoid numpy issues
    # Use direct call with model() instead of model.predict()
    results = model(pred_image_path, conf=conf_threshold, verbose=False)
    
    # Extract all the detected boxes, classes, and confidence scores
    boxes_all = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    classes_all = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
    scores_all = results[0].boxes.conf.cpu().numpy()

    # Filter by target class if specified
    if TARGET_CLASS_ID is not None:
        # Only keep detections of the target class
        filtered = [(b, s) for b, c, s in zip(boxes_all, classes_all, scores_all) if c == TARGET_CLASS_ID]
        boxes_all, scores_all = zip(*filtered) if filtered else ([], [])
    else:
        # Keep all detections as is
        boxes_all = boxes_all
        scores_all = scores_all

    # Apply Non-Maximum Suppression to remove duplicate detections
    # This helps when the model detects the same object multiple times
    if boxes_all:
        boxes_tensor = torch.tensor(boxes_all, dtype=torch.float32)
        scores_tensor = torch.tensor(scores_all, dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        boxes_all = [boxes_all[i] for i in keep]
        scores_all = [scores_all[i] for i in keep]

    # Add padding to predicted boxes (helps with slight misalignments)
    # But make sure we don't go outside the image boundaries
    pred_boxes_padded = []
    for box in boxes_all:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad_pixels)           # Don't go below 0
        y1 = max(0, y1 - pad_pixels)           # Don't go below 0
        x2 = min(img_w, x2 + pad_pixels)       # Don't exceed image width
        y2 = min(img_h, y2 + pad_pixels)       # Don't exceed image height
        pred_boxes_padded.append([x1, y1, x2, y2])

    # Load ground truth boxes from JSON annotation file
    gt_boxes = extract_boxes_from_json(gt_json_path)
    print(f"[DEBUG] GT boxes: {len(gt_boxes)}, Predicted boxes: {len(pred_boxes_padded)}")

    # Sort predictions by confidence score (highest first)
    # This is important for calculating precision-recall curve correctly
    if scores_all:
        sorted_indices = np.argsort(-np.array(scores_all))
        pred_boxes_sorted = [pred_boxes_padded[i] for i in sorted_indices]
        scores_sorted = [scores_all[i] for i in sorted_indices]
    else:
        pred_boxes_sorted = []
        scores_sorted = []

    # Match predictions with ground truth boxes
    matched_gt = set()  # Keep track of which GT boxes have been matched
    tp_list, fp_list, ious = [], [], []

    # Go through each prediction (in order of confidence)
    for pb in pred_boxes_sorted:
        match_found = False
        
        # Try to match with a ground truth box
        for j, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            
            # If IoU is good enough and this GT hasn't been matched yet
            if iou >= iou_threshold and j not in matched_gt:
                matched_gt.add(j)  # Mark this GT as matched
                ious.append(iou)   # Store the IoU for statistics
                match_found = True
                break
        
        # Record if this prediction was a true positive or false positive
        tp_list.append(1 if match_found else 0)
        fp_list.append(0 if match_found else 1)

    # Calculate False Negatives (GT boxes that weren't detected)
    FN = len(gt_boxes) - len(matched_gt)
    TP = sum(tp_list)
    FP = sum(fp_list)

    # Calculate all our evaluation metrics
    precision = TP / (TP + FP + 1e-6)                    # How many detections were correct?
    recall = TP / (TP + FN + 1e-6)                       # How many GT boxes did we find?
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)  # Harmonic mean
    mean_iou = np.mean(ious) if ious else 0.0            # Average overlap quality
    accuracy = TP / (TP + FP + FN + 1e-6)                # Simple accuracy metric

    # Build precision-recall curve for Average Precision calculation
    tp_cumsum = np.cumsum(tp_list)      # Running total of true positives
    fp_cumsum = np.cumsum(fp_list)      # Running total of false positives
    precisions_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls_curve = tp_cumsum / (len(gt_boxes) + 1e-6)
    
    # Calculate Average Precision (area under PR curve)
    ap = calculate_average_precision(recalls_curve, precisions_curve)

    # Print all the results in a nice format
    print("\n" + "="*70)
    print("[INFO] EVALUATION METRICS")
    print("="*70)
    print(f"True Positives (TP)      : {TP}")
    print(f"False Positives (FP)     : {FP}")
    print(f"False Negatives (FN)     : {FN}")
    print(f"Precision                : {precision:.4f}")
    print(f"Recall                   : {recall:.4f}")
    print(f"F1 Score                 : {f1_score:.4f}")
    print(f"Mean IoU                 : {mean_iou:.4f}")
    print(f"Accuracy (simplified)    : {accuracy:.4f}")
    print(f"Average Precision (AP)   : {ap:.4f}")
    print("="*70 + "\n")

    # Plot the precision-recall curve if requested
    if plot_pr_curve and len(precisions_curve) > 0:
        plot_precision_recall_curve(
            precisions_curve, 
            recalls_curve, 
            ap, 
            "Hardik_Final/precision_recall_curve.png"
        )
    
    # Analyze different confidence thresholds if requested
    # This is what your teacher wants - to see how precision/recall changes with threshold
    if analyze_thresholds:
        print("\n" + "="*70)
        print("[INFO] ANALYZING PRECISION-RECALL TRADE-OFF")
        print("="*70)
        threshold_results = analyze_confidence_thresholds(
            pred_image_path, gt_json_path, model_path, 
            iou_threshold=iou_threshold, pad_pixels=pad_pixels
        )
        
        # Plot threshold analysis
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Precision and Recall vs Threshold
        plt.subplot(1, 2, 1)
        thresholds = [r['threshold'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1_scores = [r['f1'] for r in threshold_results]
        
        plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g--', label='F1 Score', linewidth=2)
        
        # Mark important points
        # Best F1 point
        best_f1_idx = np.argmax(f1_scores)
        plt.plot(thresholds[best_f1_idx], f1_scores[best_f1_idx], 'go', markersize=10, label='Best F1')
        
        # High recall point
        for i, r in enumerate(recalls):
            if r >= 0.9:
                plt.plot(thresholds[i], recalls[i], 'ro', markersize=10, label='90% Recall')
                break
        
        plt.xlabel('Confidence Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Confidence Threshold\n(For Chinese OCR: Lower threshold = Higher recall)', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.1, 0.9])
        plt.ylim([0, 1.05])
        
        # Subplot 2: Trade-off curve (Precision vs Recall)
        plt.subplot(1, 2, 2)
        plt.plot(recalls, precisions, 'b-o', markersize=4, linewidth=2)
        
        # Annotate some key points
        for i in range(0, len(thresholds), 3):  # Every 3rd point
            plt.annotate(f'{thresholds[i]:.2f}', 
                        xy=(recalls[i], precisions[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        # Highlight the trade-off zone for Chinese OCR
        plt.axvspan(0.85, 1.0, alpha=0.2, color='green', label='High Recall Zone\n(Recommended for Chinese OCR)')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Trade-off\n(numbers show confidence thresholds)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        plt.savefig("Hardik_Final/threshold_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Threshold analysis plot saved to Hardik_Final/threshold_analysis.png")
        
        # Create an additional plot specifically for your teacher
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=3)
        plt.fill_between(recalls, precisions, alpha=0.3)
        
        # Mark key operating points
        for i, threshold in enumerate([0.2, 0.3, 0.5, 0.7]):
            idx = next((j for j, r in enumerate(threshold_results) if r['threshold'] >= threshold), -1)
            if idx >= 0:
                r = threshold_results[idx]
                plt.plot(r['recall'], r['precision'], 'o', markersize=10)
                plt.annotate(f"conf={threshold:.1f}\nP={r['precision']:.2f}\nR={r['recall']:.2f}", 
                           xy=(r['recall'], r['precision']),
                           xytext=(10, -20), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.xlabel('Recall (Character Detection Rate)', fontsize=14)
        plt.ylabel('Precision (Detection Accuracy)', fontsize=14)
        plt.title('Precision-Recall Trade-off for Chinese Character Detection', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig("Hardik_Final/chinese_ocr_tradeoff.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Chinese OCR trade-off analysis saved to Hardik_Final/chinese_ocr_tradeoff.png")

    # Create visualization to see how well we did
    vis_img = img.copy()
    
    # Draw predicted boxes (green = correct, red = wrong)
    for i, pb in enumerate(pred_boxes_sorted):
        # Choose color based on whether this was a true positive
        color = (0, 255, 0) if tp_list[i] == 1 else (0, 0, 255)  # Green or Red
        x1, y1, x2, y2 = pb
        conf = scores_sorted[i]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Add confidence score as label
        label = f"{conf:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw ground truth boxes in blue for reference
    for gb in gt_boxes:
        x1, y1, x2, y2 = gb
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color

    # Save the visualization so we can see how we did
    cv2.imwrite("Hardik_Final/result_debug.jpg", vis_img)
    print(f"[INFO] Visualization saved to Hardik_Final/result_debug.jpg")

if __name__ == "__main__":
    # Run evaluation on a specific image
    print("\n" + "="*70)
    print("CHINESE CHARACTER DETECTION EVALUATION")
    print("="*70)
    
    evaluate_prediction(
        pred_image_path='LabeledImage/8_org.jpeg',
        gt_json_path='LabeledImage/8_org.jpeg.json',
        model_path='best.pt',
        iou_threshold=0.1,     # Pretty lenient - accepts boxes with just 10% overlap
        conf_threshold=0.5,    # Only accept detections with 50%+ confidence
        pad_pixels=4,          # Add small padding to help with slight misalignments
        plot_pr_curve=True,    # Generate precision-recall curve
        analyze_thresholds=True  # Analyze different confidence thresholds
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("Check 'Hardik_Final' folder for:")
    print("  1. precision_recall_curve.png - Overall PR curve")
    print("  2. threshold_analysis.png - Detailed threshold analysis")
    print("  3. chinese_ocr_tradeoff.png - Specific Chinese OCR recommendations")
    print("  4. result_debug.jpg - Visual detection results")
    print("="*70)
