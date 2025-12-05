#!/usr/bin/env python3
"""
Extract small patches from COCO-format training data for binary classification.
Creates balanced dataset of caravan and non-caravan patches.

Based on ONS methodology: https://www.ons.gov.uk/methodology/methodologicalpublications
"""

import json
import os
import random
import argparse
from PIL import Image
import numpy as np
from pathlib import Path


def get_polygon_centroid(segmentation):
    """Get centroid of a polygon annotation."""
    if not segmentation or not segmentation[0]:
        return None

    seg = segmentation[0]
    xs = [seg[i] for i in range(0, len(seg), 2)]
    ys = [seg[i] for i in range(1, len(seg), 2)]

    return (sum(xs) / len(xs), sum(ys) / len(ys))


def get_bbox_centroid(bbox):
    """Get centroid of a bounding box [x, y, w, h]."""
    x, y, w, h = bbox
    return (x + w/2, y + h/2)


def extract_patch(image, center_x, center_y, patch_size):
    """Extract a square patch centered at (center_x, center_y)."""
    half = patch_size // 2

    # Calculate bounds
    left = int(center_x - half)
    top = int(center_y - half)
    right = left + patch_size
    bottom = top + patch_size

    # Check bounds
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        return None

    return image.crop((left, top, right, bottom))


def is_overlap(x, y, occupied_regions, min_distance):
    """Check if point overlaps with any occupied region."""
    for ox, oy in occupied_regions:
        if abs(x - ox) < min_distance and abs(y - oy) < min_distance:
            return True
    return False


def extract_patches_from_dataset(data_dir, output_dir, patch_size=48,
                                  patches_per_caravan=4, negative_ratio=1.0,
                                  max_images=None):
    """
    Extract caravan and non-caravan patches from COCO dataset.

    Args:
        data_dir: Directory containing images/ and annotations.json
        output_dir: Output directory for patches
        patch_size: Size of square patches to extract
        patches_per_caravan: Number of augmented patches per caravan (with jitter)
        negative_ratio: Ratio of negative to positive patches
        max_images: Maximum number of images to process (None for all)
    """
    # Load annotations
    ann_path = os.path.join(data_dir, 'annotations.json')
    with open(ann_path) as f:
        coco = json.load(f)

    # Create output directories
    pos_dir = os.path.join(output_dir, 'caravan')
    neg_dir = os.path.join(output_dir, 'not_caravan')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Build lookup
    images_by_id = {img['id']: img for img in coco['images']}
    anns_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    print(f"Dataset: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Patches per caravan: {patches_per_caravan}")

    pos_count = 0
    neg_count = 0

    image_ids = list(images_by_id.keys())
    if max_images:
        image_ids = image_ids[:max_images]

    for idx, img_id in enumerate(image_ids):
        img_info = images_by_id[img_id]
        img_path = os.path.join(data_dir, 'images', img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB')
        annotations = anns_by_image.get(img_id, [])

        if not annotations:
            continue

        # Track caravan locations for negative sampling
        caravan_locations = []

        # Extract positive patches (caravans)
        for ann in annotations:
            # Get centroid
            if 'segmentation' in ann and ann['segmentation']:
                centroid = get_polygon_centroid(ann['segmentation'])
            elif 'bbox' in ann:
                centroid = get_bbox_centroid(ann['bbox'])
            else:
                continue

            if centroid is None:
                continue

            cx, cy = centroid
            caravan_locations.append((cx, cy))

            # Extract patch with small random jitter for augmentation
            for j in range(patches_per_caravan):
                if j == 0:
                    jx, jy = 0, 0  # First patch is centered
                else:
                    # Add small jitter (up to 25% of patch size)
                    jitter = patch_size // 4
                    jx = random.randint(-jitter, jitter)
                    jy = random.randint(-jitter, jitter)

                patch = extract_patch(image, cx + jx, cy + jy, patch_size)
                if patch:
                    patch_name = f"pos_{img_id}_{ann['id']}_{j}.png"
                    patch.save(os.path.join(pos_dir, patch_name))
                    pos_count += 1

        # Extract negative patches (not caravans)
        num_negatives = int(len(annotations) * patches_per_caravan * negative_ratio)
        neg_extracted = 0
        attempts = 0
        max_attempts = num_negatives * 20

        while neg_extracted < num_negatives and attempts < max_attempts:
            attempts += 1

            # Random location
            half = patch_size // 2
            x = random.randint(half, image.width - half)
            y = random.randint(half, image.height - half)

            # Check not overlapping with any caravan
            min_distance = patch_size * 1.5  # Ensure good separation
            if not is_overlap(x, y, caravan_locations, min_distance):
                patch = extract_patch(image, x, y, patch_size)
                if patch:
                    patch_name = f"neg_{img_id}_{neg_extracted}.png"
                    patch.save(os.path.join(neg_dir, patch_name))
                    neg_count += 1
                    neg_extracted += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_ids)} images...")

    print(f"\nExtraction complete!")
    print(f"  Caravan patches: {pos_count}")
    print(f"  Non-caravan patches: {neg_count}")
    print(f"  Total: {pos_count + neg_count}")
    print(f"\nOutput saved to: {output_dir}")

    # Create a simple metadata file
    metadata = {
        'patch_size': patch_size,
        'num_positive': pos_count,
        'num_negative': neg_count,
        'source_dir': data_dir
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return pos_count, neg_count


def main():
    parser = argparse.ArgumentParser(description='Extract patches for binary classification')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing images/ and annotations.json')
    parser.add_argument('--output-dir', type=str, default='patches',
                        help='Output directory for patches')
    parser.add_argument('--patch-size', type=int, default=48,
                        help='Size of square patches (default: 48)')
    parser.add_argument('--patches-per-caravan', type=int, default=4,
                        help='Number of patches per caravan with jitter (default: 4)')
    parser.add_argument('--negative-ratio', type=float, default=1.0,
                        help='Ratio of negative to positive patches (default: 1.0)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum images to process (default: all)')
    args = parser.parse_args()

    extract_patches_from_dataset(
        args.data_dir,
        args.output_dir,
        patch_size=args.patch_size,
        patches_per_caravan=args.patches_per_caravan,
        negative_ratio=args.negative_ratio,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
