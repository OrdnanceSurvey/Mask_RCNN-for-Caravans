#!/usr/bin/env python3
"""Visualize COCO annotations overlaid on images to verify alignment."""

import json
import os
import random
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_coco_annotations(json_path):
    """Load COCO format annotations."""
    with open(json_path, 'r') as f:
        return json.load(f)


def visualize_image(image_path, annotations, output_path=None, show=True):
    """Draw annotations on an image."""
    img = Image.open(image_path).convert('RGBA')

    # Create overlay for semi-transparent masks
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw each annotation
    for ann in annotations:
        # Random color for each instance
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

        # Draw polygon mask
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) >= 6:  # Need at least 3 points
                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    # Fill with semi-transparent color
                    draw.polygon(points, fill=color + (100,), outline=color + (255,))

        # Draw bounding box
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x+w, y+h], outline=color + (255,), width=2)

    # Composite overlay on image
    result = Image.alpha_composite(img, overlay)

    if output_path:
        result.convert('RGB').save(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(result)
        plt.axis('off')
        plt.title(os.path.basename(image_path))
        plt.tight_layout()
        plt.show()

    return result


def main():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations')
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Directory containing images/ and annotations.json')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random images to visualize')
    parser.add_argument('--image-id', type=int, default=None,
                        help='Specific image ID to visualize')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save visualizations (optional)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display images (just save)')
    args = parser.parse_args()

    # Load annotations
    ann_path = os.path.join(args.data_dir, 'annotations.json')
    if not os.path.exists(ann_path):
        print(f"Error: {ann_path} not found")
        return

    coco = load_coco_annotations(ann_path)

    # Build lookup dictionaries
    images_by_id = {img['id']: img for img in coco['images']}
    anns_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    print(f"Loaded {len(coco['images'])} images with {len(coco['annotations'])} annotations")

    # Select images to visualize
    if args.image_id is not None:
        if args.image_id in images_by_id:
            image_ids = [args.image_id]
        else:
            print(f"Image ID {args.image_id} not found")
            return
    else:
        image_ids = list(images_by_id.keys())
        random.shuffle(image_ids)
        image_ids = image_ids[:args.num_samples]

    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Visualize each image
    for img_id in image_ids:
        img_info = images_by_id[img_id]
        img_path = os.path.join(args.data_dir, 'images', img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue

        annotations = anns_by_image.get(img_id, [])
        print(f"\nImage: {img_info['file_name']} ({len(annotations)} annotations)")

        output_path = None
        if args.save_dir:
            output_path = os.path.join(args.save_dir, f"viz_{img_info['file_name']}")

        visualize_image(img_path, annotations, output_path=output_path, show=not args.no_show)


if __name__ == '__main__':
    main()
