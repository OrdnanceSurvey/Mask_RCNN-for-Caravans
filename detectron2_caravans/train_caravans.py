"""
Detectron2 Caravan Detection Training Script

Train Mask R-CNN to detect caravans in satellite/aerial imagery.

Usage:
    # Train from COCO pre-trained weights
    python train_caravans.py train --dataset /path/to/dataset

    # Resume training from checkpoint
    python train_caravans.py train --dataset /path/to/dataset --weights /path/to/model.pth

    # Run inference on an image
    python train_caravans.py detect --weights /path/to/model.pth --image /path/to/image.tif

    # Run inference on a folder of images
    python train_caravans.py detect --weights /path/to/model.pth --folder /path/to/images/

Dataset structure expected:
    dataset/
    ├── train/
    │   ├── images/
    │   │   └── *.tif
    │   └── labels/
    │       └── 1/
    │           └── *.png  (instance masks)
    └── val/
        ├── images/
        │   └── *.tif
        └── labels/
            └── 1/
                └── *.png
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path

import torch
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator


def get_caravan_dicts(dataset_dir, subset):
    """
    Load caravan dataset in Detectron2 format.

    Args:
        dataset_dir: Root directory of the dataset
        subset: 'train' or 'val'

    Returns:
        List of dictionaries, one per image
    """
    subset_dir = os.path.join(dataset_dir, subset)
    image_dir = os.path.join(subset_dir, "images")
    mask_dir = os.path.join(subset_dir, "labels", "1")

    dataset_dicts = []

    # Find all images
    image_files = glob.glob(os.path.join(image_dir, "*.tif"))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.png")))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))

    for idx, image_path in enumerate(image_files):
        record = {}

        # Read image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read {image_path}, skipping...")
            continue

        height, width = image.shape[:2]

        record["file_name"] = image_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Find corresponding mask file
        image_name = os.path.basename(image_path)
        mask_name = image_name.replace('.tif', '.png').replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)

        annotations = []

        if os.path.exists(mask_path):
            # Read instance mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if mask is not None:
                # Get unique instance IDs (excluding 0 which is background)
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]

                for instance_id in instance_ids:
                    # Create binary mask for this instance
                    binary_mask = (mask == instance_id).astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if len(contours) == 0:
                        continue

                    # Get the largest contour
                    contour = max(contours, key=cv2.contourArea)

                    # Skip very small instances
                    if cv2.contourArea(contour) < 10:
                        continue

                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Convert contour to polygon format
                    if len(contour) >= 3:
                        poly = contour.flatten().tolist()

                        annotation = {
                            "bbox": [x, y, x + w, y + h],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": 0,  # 0 = caravan (single class)
                            "iscrowd": 0,
                        }
                        annotations.append(annotation)

        record["annotations"] = annotations
        dataset_dicts.append(record)

    print(f"Loaded {len(dataset_dicts)} images from {subset} set")
    return dataset_dicts


def register_caravan_dataset(dataset_dir):
    """Register the caravan dataset with Detectron2."""

    for subset in ["train", "val"]:
        dataset_name = f"caravan_{subset}"

        # Remove if already registered
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)

        # Register dataset
        DatasetCatalog.register(
            dataset_name,
            lambda d=dataset_dir, s=subset: get_caravan_dicts(d, s)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=["caravan"])

    print("Dataset registered successfully!")


class CaravanTrainer(DefaultTrainer):
    """Custom trainer with COCO evaluator."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_dir = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_dir)


def setup_cfg(args, num_classes=1):
    """
    Set up Detectron2 configuration.

    Args:
        args: Command line arguments
        num_classes: Number of classes (1 for caravan only)

    Returns:
        Detectron2 config object
    """
    cfg = get_cfg()

    # Use Mask R-CNN with ResNet-101 FPN backbone (similar to original)
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    ))

    # Dataset
    cfg.DATASETS.TRAIN = ("caravan_train",)
    cfg.DATASETS.TEST = ("caravan_val",)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4

    # Model weights
    if args.weights and os.path.exists(args.weights):
        cfg.MODEL.WEIGHTS = args.weights
        print(f"Loading weights from: {args.weights}")
    else:
        # Start from COCO pre-trained weights
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
        print("Using COCO pre-trained weights")

    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000  # Adjust based on dataset size
    cfg.SOLVER.STEPS = (7000, 9000)  # LR decay steps
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # Model settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold

    # RPN anchor scales (similar to original config)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]

    # Input image size
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Data augmentation
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Output directory
    cfg.OUTPUT_DIR = args.output if args.output else "./output_caravan"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Evaluation period
    cfg.TEST.EVAL_PERIOD = 500

    return cfg


def train(args):
    """Train the model."""
    print("=" * 60)
    print("CARAVAN DETECTION - TRAINING")
    print("=" * 60)

    # Register dataset
    register_caravan_dataset(args.dataset)

    # Setup config
    cfg = setup_cfg(args)

    # Print config summary
    print(f"\nConfiguration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Output: {cfg.OUTPUT_DIR}")
    print(f"  - Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  - Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # Create trainer and start training
    trainer = CaravanTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    print("\nTraining complete!")
    print(f"Model saved to: {cfg.OUTPUT_DIR}")


def detect(args):
    """Run detection on images."""
    print("=" * 60)
    print("CARAVAN DETECTION - INFERENCE")
    print("=" * 60)

    if not args.weights:
        print("Error: --weights required for detection")
        sys.exit(1)

    # Setup config for inference
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Set up metadata for visualization
    metadata = MetadataCatalog.get("caravan_val") if "caravan_val" in DatasetCatalog.list() else None
    if metadata is None:
        from detectron2.data import MetadataCatalog
        MetadataCatalog.get("caravan_inference").set(thing_classes=["caravan"])
        metadata = MetadataCatalog.get("caravan_inference")

    # Create output directory
    output_dir = args.output if args.output else "./detections"
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images to process
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.folder:
        for ext in ['*.tif', '*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))

    if not image_paths:
        print("No images found to process!")
        return

    print(f"Processing {len(image_paths)} images...")

    for image_path in image_paths:
        print(f"  Processing: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"    Warning: Could not read {image_path}")
            continue

        # Run detection
        outputs = predictor(image)

        # Get results
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)

        print(f"    Found {num_detections} caravan(s)")

        # Visualize results
        v = Visualizer(
            image[:, :, ::-1],  # BGR to RGB
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        vis_output = v.draw_instance_predictions(instances)

        # Save visualization
        output_path = os.path.join(
            output_dir,
            Path(image_path).stem + "_detected.png"
        )
        cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])

        # Also save mask output if requested
        if args.save_masks and num_detections > 0:
            masks = instances.pred_masks.numpy()
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for i, mask in enumerate(masks):
                combined_mask[mask] = i + 1

            mask_path = os.path.join(
                output_dir,
                Path(image_path).stem + "_mask.png"
            )
            cv2.imwrite(mask_path, combined_mask)

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and run Mask R-CNN for caravan detection"
    )
    parser.add_argument(
        "command",
        choices=["train", "detect"],
        help="'train' to train model, 'detect' to run inference"
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset directory (required for training)"
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "--output",
        help="Output directory for model/detections"
    )
    parser.add_argument(
        "--image",
        help="Single image path for detection"
    )
    parser.add_argument(
        "--folder",
        help="Folder of images for batch detection"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save instance masks during detection"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        if not args.dataset:
            parser.error("--dataset is required for training")
    elif args.command == "detect":
        if not args.image and not args.folder:
            parser.error("--image or --folder is required for detection")

    # Run command
    if args.command == "train":
        train(args)
    elif args.command == "detect":
        detect(args)


if __name__ == "__main__":
    main()
