#!/usr/bin/env python3
"""
Debug script to visualize mask alignment issues.

Usage:
    python debug_alignment.py --dataset path/to/dataset --output debug.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def debug_alignment(dataset_path, output_path, num_samples=20):
    """
    Create a debug visualization showing images with OSM polygons and masks.
    """
    dataset_path = Path(dataset_path)

    # Try to find metadata
    metadata_path = dataset_path / "imagery" / "metadata.json"
    if not metadata_path.exists():
        # Maybe they passed the training folder directly
        metadata_path = dataset_path.parent / "imagery" / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: metadata.json not found")
        print(f"Looked in: {dataset_path / 'imagery' / 'metadata.json'}")
        return

    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Total images in metadata: {len(metadata)}")

    # Find training folder
    training_path = dataset_path / "training"
    if not training_path.exists():
        training_path = dataset_path

    train_images_dir = training_path / "train" / "images"
    train_masks_dir = training_path / "train" / "labels" / "1"

    print(f"Training images: {train_images_dir}")
    print(f"Training masks: {train_masks_dir}")

    # Create figure
    samples = min(num_samples, len(metadata))
    cols = 5
    rows = (samples * 2 + cols - 1) // cols  # 2 rows per sample (image + mask overlay)

    # Calculate figure size
    fig_width = cols * 3
    fig_height = rows * 3

    # Create output image
    cell_size = 200
    output_img = Image.new('RGB', (cols * cell_size, rows * cell_size), (255, 255, 255))

    for idx in range(samples):
        item = metadata[idx]

        col = idx % cols
        row = (idx // cols) * 2  # Each sample takes 2 rows

        # === Row 1: Image with OSM polygon ===
        try:
            img = Image.open(item['image_path']).convert('RGB')
        except:
            print(f"  Could not load: {item['image_path']}")
            continue

        img_w, img_h = img.size

        # Get the feature polygon coordinates
        coords = item['feature']['geometry']['coordinates'][0]

        # Get the image bbox
        bbox = item['bbox']
        west, south, east, north = bbox

        # Convert polygon to pixel coordinates
        pixel_coords = []
        for lon, lat in coords:
            px = int((lon - west) / (east - west) * img_w)
            py = int((north - lat) / (north - south) * img_h)
            pixel_coords.append((px, py))

        # Draw polygon on image
        img_with_poly = img.copy()
        draw = ImageDraw.Draw(img_with_poly, 'RGBA')

        # Fill polygon with semi-transparent red
        if len(pixel_coords) >= 3:
            draw.polygon(pixel_coords, fill=(255, 0, 0, 100), outline=(255, 0, 0))

        # Resize for display
        img_thumb = img_with_poly.resize((cell_size, cell_size), Image.LANCZOS)
        output_img.paste(img_thumb, (col * cell_size, row * cell_size))

        # === Row 2: Training image with mask ===
        osm_id = item['feature']['properties']['osm_id']

        # Find corresponding training image and mask
        train_img_path = None
        for ext in ['.tif', '.png', '.jpg']:
            potential = train_images_dir / f"{osm_id}{ext}"
            if potential.exists():
                train_img_path = potential
                break

        if train_img_path and train_img_path.exists():
            train_img = Image.open(train_img_path).convert('RGB')

            # Find mask
            mask_path = train_masks_dir / f"{osm_id}.png"

            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask_array = np.array(mask)

                # Create overlay
                train_array = np.array(train_img)
                overlay = train_array.copy()
                overlay[mask_array > 0] = [255, 0, 0]

                # Blend
                result = (train_array * 0.7 + overlay * 0.3).astype(np.uint8)
                result_img = Image.fromarray(result)
            else:
                result_img = train_img
                # Add "NO MASK" text
                draw = ImageDraw.Draw(result_img)
                draw.text((10, 10), "NO MASK", fill=(255, 0, 0))

            # Resize for display
            result_thumb = result_img.resize((cell_size, cell_size), Image.LANCZOS)
            output_img.paste(result_thumb, (col * cell_size, (row + 1) * cell_size))

        print(f"  Processed {idx + 1}/{samples}: OSM ID {osm_id}, size {img_w}x{img_h}")

    # Save output
    output_path = Path(output_path)
    output_img.save(output_path)
    print(f"\nSaved debug image to: {output_path}")
    print("\nTop rows: Downloaded images with OSM polygon (red)")
    print("Bottom rows: Training images with mask overlay (red)")
    print("\nIf polygons don't match caravans, the bbox calculation is wrong.")
    print("If masks don't match polygons, the mask generation is wrong.")


def main():
    parser = argparse.ArgumentParser(description="Debug mask alignment")
    parser.add_argument('--dataset', required=True, help='Path to dataset folder')
    parser.add_argument('--output', default='alignment_debug.png', help='Output image path')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples to show')

    args = parser.parse_args()

    debug_alignment(args.dataset, args.output, args.samples)


if __name__ == "__main__":
    main()
