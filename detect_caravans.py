#!/usr/bin/env python3
"""
Caravan Detection Script
Detects caravans in an image and draws red bounding boxes around them.

Usage:
    python detect_caravans.py --image <path_to_image> --weights <path_to_weights>

Example:
    python detect_caravans.py --image test.jpg --weights logs/caravan20201231T1234/mask_rcnn_caravan_0050.h5
"""

import os
import sys
import argparse
import numpy as np
import cv2
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib


class CaravanConfig(Config):
    """Configuration for caravan inference."""
    NAME = "caravan"

    # Run on 1 GPU with 1 image at a time for inference
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (background + caravan)
    NUM_CLASSES = 1 + 1

    # Detection confidence threshold
    DETECTION_MIN_CONFIDENCE = 0.9

    # Backbone architecture
    BACKBONE = "resnet101"

    # Anchor scales
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)


def load_model(weights_path, logs_dir=None):
    """Load the Mask R-CNN model with the given weights.

    Args:
        weights_path: Path to the weights file (.h5)
        logs_dir: Directory for model logs (optional)

    Returns:
        Loaded model ready for inference
    """
    if logs_dir is None:
        logs_dir = os.path.join(ROOT_DIR, "logs")

    config = CaravanConfig()

    # Create model in inference mode
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=logs_dir
    )

    # Load weights
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path, by_name=True)

    return model


def detect_caravans(model, image_path):
    """Run caravan detection on an image.

    Args:
        model: Loaded Mask R-CNN model
        image_path: Path to the input image

    Returns:
        image: The original image as numpy array
        results: Detection results containing boxes, masks, class_ids, scores
    """
    # Read image
    image = skimage.io.imread(image_path)

    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Handle RGBA images
    if image.shape[-1] == 4:
        image = image[..., :3]

    print(f"Running detection on: {image_path}")
    print(f"Image shape: {image.shape}")

    # Run detection
    results = model.detect([image], verbose=1)[0]

    return image, results


def draw_red_boxes(image, boxes, scores=None, thickness=3):
    """Draw red bounding boxes on the image.

    Args:
        image: Input image (numpy array)
        boxes: Array of bounding boxes [N, (y1, x1, y2, x2)]
        scores: Optional confidence scores for each box
        thickness: Line thickness for boxes (default: 3)

    Returns:
        Image with red bounding boxes drawn
    """
    # Make a copy to avoid modifying original
    output = image.copy()

    # Ensure image is in correct format for OpenCV (BGR)
    if output.dtype != np.uint8:
        output = output.astype(np.uint8)

    # Red color in BGR format for OpenCV
    RED = (0, 0, 255)

    num_detections = boxes.shape[0]
    print(f"Drawing {num_detections} bounding box(es)")

    for i in range(num_detections):
        # Get box coordinates (y1, x1, y2, x2)
        y1, x1, y2, x2 = boxes[i]

        # Convert to integers
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

        # Draw rectangle (OpenCV uses (x1, y1), (x2, y2) format)
        cv2.rectangle(output, (x1, y1), (x2, y2), RED, thickness)

        # Add confidence score label if available
        if scores is not None:
            score = scores[i]
            label = f"Caravan: {score:.2f}"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                output,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                RED,
                -1  # Filled
            )

            # Draw text in white
            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )

    return output


def save_output(image, output_path):
    """Save the output image.

    Args:
        image: Image to save (RGB format)
        output_path: Path to save the image
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect caravans in an image and draw red bounding boxes."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained weights file (.h5)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output image (default: <input>_detected.<ext>)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Minimum detection confidence (default: 0.9)"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Bounding box line thickness (default: 3)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show confidence score labels on boxes"
    )

    args = parser.parse_args()

    # Validate input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Validate weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    # Generate output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_detected{ext}"

    # Load model
    print("Loading model...")
    model = load_model(args.weights)

    # Update confidence threshold if specified
    if args.confidence != 0.9:
        model.config.DETECTION_MIN_CONFIDENCE = args.confidence

    # Run detection
    image, results = detect_caravans(model, args.image)

    # Get detection results
    boxes = results["rois"]
    scores = results["scores"]
    class_ids = results["class_ids"]

    # Filter to only caravan detections (class_id = 1)
    caravan_mask = class_ids == 1
    caravan_boxes = boxes[caravan_mask]
    caravan_scores = scores[caravan_mask]

    print(f"\nDetected {len(caravan_boxes)} caravan(s)")

    if len(caravan_boxes) == 0:
        print("No caravans detected in the image.")
        # Still save the original image
        save_output(image, args.output)
    else:
        # Draw red boxes
        scores_for_drawing = None if args.no_labels else caravan_scores
        output_image = draw_red_boxes(
            image,
            caravan_boxes,
            scores=scores_for_drawing,
            thickness=args.thickness
        )

        # Save output
        save_output(output_image, args.output)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
