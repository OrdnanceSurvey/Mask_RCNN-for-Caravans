#!/usr/bin/env python3
"""
Sliding Window Inference for Caravan Detection

Since the model is trained on small tight crops (~256px), we need to use
sliding window inference on large aerial images to detect caravans at the
correct scale.

Usage:
    python sliding_window_inference.py --image path/to/aerial.jpg --model path/to/model_final.pth --output result.png
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances


def setup_cfg(model_path, score_threshold=0.5):
    """Set up Detectron2 config for inference."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # caravan
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

    # Use CPU if no GPU available
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    return cfg


def sliding_window(image, window_size=256, overlap=0.25):
    """
    Generate sliding window coordinates over an image.

    Args:
        image: numpy array (H, W, C)
        window_size: size of each window (square)
        overlap: overlap fraction between windows (0.25 = 25% overlap)

    Yields:
        (x, y, window) tuples where x, y are top-left coordinates
    """
    h, w = image.shape[:2]
    step = int(window_size * (1 - overlap))

    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            window = image[y:y+window_size, x:x+window_size]
            yield x, y, window

    # Handle right edge
    if w % step != 0:
        for y in range(0, h - window_size + 1, step):
            x = w - window_size
            window = image[y:y+window_size, x:x+window_size]
            yield x, y, window

    # Handle bottom edge
    if h % step != 0:
        for x in range(0, w - window_size + 1, step):
            y = h - window_size
            window = image[y:y+window_size, x:x+window_size]
            yield x, y, window

    # Handle bottom-right corner
    if w % step != 0 and h % step != 0:
        x, y = w - window_size, h - window_size
        window = image[y:y+window_size, x:x+window_size]
        yield x, y, window


def nms_boxes(boxes, scores, masks, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.

    Args:
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        scores: numpy array of shape (N,)
        masks: list of masks
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered boxes, scores, and masks
    """
    if len(boxes) == 0:
        return boxes, scores, masks

    # Convert to torch tensors for NMS
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # Apply NMS
    from torchvision.ops import nms
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    keep_indices = keep_indices.numpy()

    return boxes[keep_indices], scores[keep_indices], [masks[i] for i in keep_indices]


def run_sliding_window_inference(image_path, model_path, output_path,
                                  window_size=256, overlap=0.25,
                                  score_threshold=0.5, nms_threshold=0.3,
                                  verbose=True):
    """
    Run inference on a large image using sliding windows.

    Args:
        image_path: Path to input image
        model_path: Path to trained model weights
        output_path: Path to save result image
        window_size: Size of sliding window (should match training size)
        overlap: Overlap between windows (0.25 = 25%)
        score_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        verbose: Print progress

    Returns:
        dict with detection results
    """
    if verbose:
        print(f"Loading model from: {model_path}")

    # Set up predictor
    cfg = setup_cfg(model_path, score_threshold)
    predictor = DefaultPredictor(cfg)

    # Register metadata for visualization
    if "caravan_inference" not in MetadataCatalog.list():
        MetadataCatalog.get("caravan_inference").set(thing_classes=["caravan"])
    metadata = MetadataCatalog.get("caravan_inference")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    if verbose:
        print(f"Image size: {w}x{h}")
        print(f"Window size: {window_size}, Overlap: {overlap*100}%")

    # Collect all detections
    all_boxes = []
    all_scores = []
    all_masks = []

    # Count windows for progress
    windows = list(sliding_window(image, window_size, overlap))
    total_windows = len(windows)

    if verbose:
        print(f"Processing {total_windows} windows...")

    for i, (x, y, window) in enumerate(windows):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Window {i+1}/{total_windows}")

        # Run inference on window
        outputs = predictor(window)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            continue

        # Get detections
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        # Offset boxes to full image coordinates
        boxes[:, 0] += x  # x1
        boxes[:, 2] += x  # x2
        boxes[:, 1] += y  # y1
        boxes[:, 3] += y  # y2

        # Create full-size masks and place window masks in correct position
        for j, mask in enumerate(masks):
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y:y+window_size, x:x+window_size] = mask.astype(np.uint8)
            all_masks.append(full_mask)
            all_boxes.append(boxes[j])
            all_scores.append(scores[j])

    if verbose:
        print(f"Found {len(all_boxes)} raw detections")

    # Convert to numpy arrays
    if len(all_boxes) > 0:
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)

        # Apply NMS to remove duplicates
        boxes, scores, masks = nms_boxes(all_boxes, all_scores, all_masks, nms_threshold)

        if verbose:
            print(f"After NMS: {len(boxes)} detections")
    else:
        boxes, scores, masks = np.array([]), np.array([]), []

    # Visualize results
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create instances for visualization
    if len(boxes) > 0:
        instances = Instances((h, w))
        instances.pred_boxes = Boxes(torch.tensor(boxes))
        instances.scores = torch.tensor(scores)
        instances.pred_classes = torch.zeros(len(boxes), dtype=torch.int64)
        instances.pred_masks = torch.tensor(np.array(masks), dtype=torch.bool)

        visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(instances)
        result_image = vis_output.get_image()
    else:
        result_image = image_rgb

    # Add title
    result_pil = Image.fromarray(result_image)

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_pil.save(output_path)

    if verbose:
        print(f"\nDetected {len(boxes)} caravan(s)")
        print(f"Result saved to: {output_path}")

    # Return results
    results = {
        "num_detections": len(boxes),
        "boxes": boxes.tolist() if len(boxes) > 0 else [],
        "scores": scores.tolist() if len(scores) > 0 else [],
        "output_path": str(output_path)
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run sliding window inference for caravan detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python sliding_window_inference.py --image aerial.jpg --model model_final.pth

    # Custom window size and overlap
    python sliding_window_inference.py --image aerial.jpg --model model_final.pth --window 256 --overlap 0.5

    # Adjust thresholds
    python sliding_window_inference.py --image aerial.jpg --model model_final.pth --score-threshold 0.3 --nms-threshold 0.4
        """
    )

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to model weights (model_final.pth)")
    parser.add_argument("--output", default=None, help="Output path (default: input_detections.png)")
    parser.add_argument("--window", type=int, default=256, help="Window size in pixels (default: 256)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Window overlap fraction (default: 0.25)")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="NMS IoU threshold (default: 0.3)")
    parser.add_argument("--json", action="store_true", help="Also output results as JSON")

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        input_path = Path(args.image)
        args.output = input_path.parent / f"{input_path.stem}_detections.png"

    # Run inference
    results = run_sliding_window_inference(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        window_size=args.window,
        overlap=args.overlap,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold
    )

    # Save JSON if requested
    if args.json:
        json_path = Path(args.output).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
