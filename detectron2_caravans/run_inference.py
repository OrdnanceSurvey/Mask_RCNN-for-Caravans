#!/usr/bin/env python3
"""
Caravan Detection Inference Script

Run the trained model on new images to detect caravans.

Usage:
    # Single image
    python run_inference.py --model model_final.pth --image test.tif --output results/

    # Folder of images
    python run_inference.py --model model_final.pth --folder images/ --output results/

    # Adjust confidence threshold
    python run_inference.py --model model_final.pth --image test.tif --threshold 0.3

    # Save masks as separate files
    python run_inference.py --model model_final.pth --folder images/ --output results/ --save-masks
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

# Check if detectron2 is available
try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
except ImportError:
    print("Error: detectron2 is not installed.")
    print("Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    sys.exit(1)


def setup_predictor(model_path, threshold=0.5):
    """
    Set up the Detectron2 predictor with the trained model.

    Args:
        model_path: Path to model_final.pth
        threshold: Detection confidence threshold

    Returns:
        DefaultPredictor instance
    """
    cfg = get_cfg()

    # Use the same config as training
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # Model settings
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Just "caravan"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Use CPU if no GPU
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        print("Running on CPU (slower)")
    else:
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # Register metadata for visualization
    if "caravan_inference" not in MetadataCatalog.list():
        MetadataCatalog.get("caravan_inference").set(thing_classes=["caravan"])

    return DefaultPredictor(cfg)


def detect_image(predictor, image_path, output_dir, save_masks=False):
    """
    Run detection on a single image.

    Args:
        predictor: Detectron2 predictor
        image_path: Path to input image
        output_dir: Directory to save results
        save_masks: Whether to save instance masks

    Returns:
        dict with detection results
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Error: Could not read {image_path}")
        return None

    # Run detection
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    num_detections = len(instances)

    # Get metadata
    metadata = MetadataCatalog.get("caravan_inference")

    # Create visualization
    v = Visualizer(
        image[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )
    vis_output = v.draw_instance_predictions(instances)

    # Save visualization
    output_name = Path(image_path).stem + "_detected.png"
    output_path = Path(output_dir) / output_name
    cv2.imwrite(str(output_path), vis_output.get_image()[:, :, ::-1])

    # Save masks if requested
    if save_masks and num_detections > 0:
        masks = instances.pred_masks.numpy()

        # Combined mask (all instances)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, mask in enumerate(masks):
            combined_mask[mask] = i + 1

        mask_path = Path(output_dir) / (Path(image_path).stem + "_mask.png")
        cv2.imwrite(str(mask_path), combined_mask)

    # Collect results
    results = {
        "image": str(image_path),
        "num_detections": num_detections,
        "output": str(output_path),
    }

    if num_detections > 0:
        results["scores"] = instances.scores.numpy().tolist()
        results["boxes"] = instances.pred_boxes.tensor.numpy().tolist()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run caravan detection on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --model model_final.pth --image test.tif
  python run_inference.py --model model_final.pth --folder images/ --output results/
  python run_inference.py --model model_final.pth --image test.tif --threshold 0.3 --save-masks
        """
    )

    parser.add_argument(
        "--model", required=True,
        help="Path to trained model (model_final.pth)"
    )
    parser.add_argument(
        "--image",
        help="Path to a single image"
    )
    parser.add_argument(
        "--folder",
        help="Path to folder of images"
    )
    parser.add_argument(
        "--output", default="./detection_results",
        help="Output directory for results (default: ./detection_results)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Save instance masks as separate files"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CARAVAN DETECTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_dir}")
    print()

    # Setup predictor
    print("Loading model...")
    predictor = setup_predictor(args.model, args.threshold)
    print("Model loaded!\n")

    # Get list of images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.folder:
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg',
                    '*.TIF', '*.TIFF', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...\n")

    # Process images
    total_detections = 0
    results = []

    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}", end=" ")

        result = detect_image(predictor, image_path, output_dir, args.save_masks)

        if result:
            num = result["num_detections"]
            total_detections += num
            print(f"-> {num} caravan(s) detected")
            results.append(result)
        else:
            print("-> FAILED")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(results)}")
    print(f"Total caravans detected: {total_detections}")
    print(f"Average per image: {total_detections / max(len(results), 1):.1f}")
    print(f"\nResults saved to: {output_dir}")

    # Save results JSON
    import json
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detection data saved to: {results_path}")


if __name__ == "__main__":
    main()
