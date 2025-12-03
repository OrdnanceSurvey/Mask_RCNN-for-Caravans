#!/usr/bin/env python3
"""
Generate Training Dataset from Fully-Labeled OSM Parks

Creates training data from caravan parks where ALL caravans are labeled in OSM.
Outputs tiles that match your inference setup (default: zoom 18, 2048x2048).

This gives you:
- Training images at the SAME scale as inference
- ALL caravans labeled (no false negatives)
- COCO format ready for Detectron2

Usage:
    # Generate dataset from a region
    python generate_training_from_osm.py --region anglesey --output training_data/

    # Specify tile size and zoom to match your inference
    python generate_training_from_osm.py --region cornwall --zoom 18 --tile-size 2048 --output training_data/

    # Download specific parks by way ID
    python generate_training_from_osm.py --way-ids 123456,789012 --output training_data/
"""

import argparse
import requests
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime
import time
import random

# UK regions
UK_REGIONS = {
    "anglesey": "-4.7,53.1,-4.0,53.5",
    "north_wales_coast": "-4.0,53.1,-3.3,53.4",
    "pembrokeshire": "-5.3,51.6,-4.7,52.1",
    "cornwall": "-5.8,49.9,-4.5,50.7",
    "devon_south": "-4.2,50.2,-3.4,50.7",
    "dorset": "-2.9,50.5,-1.8,50.9",
    "norfolk": "0.5,52.5,1.8,53.0",
    "suffolk": "1.2,51.9,1.8,52.5",
    "essex": "0.5,51.5,1.2,51.9",
    "kent": "0.8,50.9,1.4,51.4",
    "yorkshire_coast": "-0.5,53.6,0.2,54.2",
    "lincolnshire": "-0.2,53.0,0.4,53.6",
    "lancashire": "-3.2,53.5,-2.8,54.0",
    "cumbria": "-3.6,54.0,-3.0,54.5",
    "wales": "-5.5,51.3,-2.6,53.5",
}


def query_osm(bbox=None, way_id=None, verbose=True):
    """Query OSM for caravan parks and caravans."""
    if way_id:
        query = f"""
        [out:json][timeout:60];
        way({way_id});
        (._;>;);
        out body;
        way({way_id});
        (way["building"="static_caravan"](around:500););
        (._;>;);
        out body;
        """
    else:
        west, south, east, north = map(float, bbox.split(","))
        query = f"""
        [out:json][timeout:90];
        (
            way["tourism"="caravan_site"]({south},{west},{north},{east});
            way["building"="static_caravan"]({south},{west},{north},{east});
        );
        (._;>;);
        out body;
        """

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    for server in servers:
        try:
            if verbose:
                print(f"Querying {server.split('/')[2]}...")
            response = requests.post(server, data={"data": query}, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            time.sleep(2)
            continue

    raise Exception("All OSM servers failed")


def parse_osm_data(data):
    """Parse OSM response into parks and caravans."""
    nodes = {}
    parks = []
    caravans = []

    for el in data.get('elements', []):
        if el['type'] == 'node':
            nodes[el['id']] = (el['lon'], el['lat'])
        elif el['type'] == 'way':
            tags = el.get('tags', {})
            coords = [nodes[n] for n in el.get('nodes', []) if n in nodes]
            if len(coords) < 3:
                continue

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]

            item = {
                'id': el['id'],
                'name': tags.get('name', 'Unnamed'),
                'coords': coords,
                'bbox': (min(lons), min(lats), max(lons), max(lats)),
                'center': (sum(lons)/len(lons), sum(lats)/len(lats)),
            }

            if tags.get('tourism') == 'caravan_site':
                parks.append(item)
            elif tags.get('building') == 'static_caravan':
                caravans.append(item)

    return parks, caravans


def match_caravans_to_parks(parks, caravans, buffer=0.003):
    """Match caravans to their parent parks."""
    for park in parks:
        p_west, p_south, p_east, p_north = park['bbox']
        park['caravans'] = [
            c for c in caravans
            if (p_west - buffer <= c['center'][0] <= p_east + buffer and
                p_south - buffer <= c['center'][1] <= p_north + buffer)
        ]
        park['caravan_count'] = len(park['caravans'])
    return parks


def lonlat_to_tile(lon, lat, zoom):
    """Convert lon/lat to tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n)
    return x, y


def tile_to_lonlat(tx, ty, zoom):
    """Convert tile coordinates to lon/lat (top-left corner)."""
    n = 2 ** zoom
    lon = tx / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ty / n))))
    return lon, lat


def download_tiles(bbox, zoom, verbose=True):
    """Download tiles covering a bounding box."""
    west, south, east, north = bbox

    min_tx, max_ty = lonlat_to_tile(west, south, zoom)
    max_tx, min_ty = lonlat_to_tile(east, north, zoom)

    # Add padding
    min_tx -= 1
    min_ty -= 1
    max_tx += 1
    max_ty += 1

    url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    tiles = []
    total = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
    count = 0

    for ty in range(min_ty, max_ty + 1):
        row = []
        for tx in range(min_tx, max_tx + 1):
            count += 1
            if verbose and count % 10 == 0:
                print(f"  Downloading tile {count}/{total}")

            url = url_template.format(z=zoom, x=tx, y=ty)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                from io import BytesIO
                tile = Image.open(BytesIO(resp.content))
                row.append(tile)
            except:
                row.append(Image.new('RGB', (256, 256), (128, 128, 128)))
            time.sleep(0.05)
        tiles.append(row)

    # Stitch
    img_width = (max_tx - min_tx + 1) * 256
    img_height = (max_ty - min_ty + 1) * 256
    stitched = Image.new('RGB', (img_width, img_height))

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            stitched.paste(tile, (x * 256, y * 256))

    # Calculate image bbox
    img_west, img_north = tile_to_lonlat(min_tx, min_ty, zoom)
    img_east, img_south = tile_to_lonlat(max_tx + 1, max_ty + 1, zoom)

    return stitched, (img_west, img_south, img_east, img_north)


def geo_to_pixel(lon, lat, img_bbox, img_size):
    """Convert geo coordinates to pixel coordinates."""
    west, south, east, north = img_bbox
    img_w, img_h = img_size
    px = int((lon - west) / (east - west) * img_w)
    py = int((north - lat) / (north - south) * img_h)
    return px, py


def pixel_to_geo(px, py, img_bbox, img_size):
    """Convert pixel coordinates to geo coordinates."""
    west, south, east, north = img_bbox
    img_w, img_h = img_size
    lon = west + (px / img_w) * (east - west)
    lat = north - (py / img_h) * (north - south)
    return lon, lat


def create_tiles_from_park(park, zoom=18, tile_size=2048, overlap=256, min_tile_caravans=1, verbose=True):
    """
    Create training tiles from a park.

    Returns list of (tile_image, tile_annotations, tile_bbox)
    """
    if verbose:
        print(f"\nProcessing: {park['name']} ({park['caravan_count']} caravans)")

    # Download full park imagery with extra padding to ensure we can create full tiles
    park_bbox = park['bbox']

    # Add padding to park bbox to ensure we have enough imagery for full tiles
    west, south, east, north = park_bbox
    # Calculate approximate degrees per pixel at this zoom
    deg_per_tile = 360 / (2 ** zoom)
    padding_deg = (tile_size / 256) * deg_per_tile * 0.5  # Extra padding

    padded_bbox = (west - padding_deg, south - padding_deg, east + padding_deg, north + padding_deg)

    full_image, img_bbox = download_tiles(padded_bbox, zoom, verbose)
    img_w, img_h = full_image.size

    if verbose:
        print(f"  Full image size: {img_w}x{img_h}")

    # Skip if image is too small even with padding
    if img_w < tile_size or img_h < tile_size:
        if verbose:
            print(f"  Skipping: park image too small ({img_w}x{img_h} < {tile_size}x{tile_size})")
        return []

    # Convert all caravan polygons to pixel coordinates
    caravans_px = []
    for caravan in park['caravans']:
        pixel_coords = [geo_to_pixel(lon, lat, img_bbox, full_image.size)
                        for lon, lat in caravan['coords']]
        caravans_px.append({
            'id': caravan['id'],
            'coords': pixel_coords
        })

    # Generate tiles - ensure we only create full-size tiles
    tiles = []
    step = tile_size - overlap

    tile_idx = 0
    y = 0
    while y + tile_size <= img_h:
        x = 0
        while x + tile_size <= img_w:
            # Extract tile - guaranteed to be full size
            tile_img = full_image.crop((x, y, x + tile_size, y + tile_size))

            # Verify size (should always be correct now)
            assert tile_img.size == (tile_size, tile_size), f"Tile size mismatch: {tile_img.size}"

            # Find caravans in this tile
            tile_annotations = []
            for caravan in caravans_px:
                # Offset coordinates to tile space
                tile_coords = [(px - x, py - y) for px, py in caravan['coords']]

                # Check if any part of the caravan is in the tile
                xs = [p[0] for p in tile_coords]
                ys = [p[1] for p in tile_coords]

                if (max(xs) < 0 or min(xs) > tile_size or
                    max(ys) < 0 or min(ys) > tile_size):
                    continue  # Completely outside tile

                # Clip coordinates to tile bounds
                clipped_coords = [
                    (max(0, min(tile_size-1, px)), max(0, min(tile_size-1, py)))
                    for px, py in tile_coords
                ]

                # Calculate bounding box
                xs = [p[0] for p in clipped_coords]
                ys = [p[1] for p in clipped_coords]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

                # Skip if too small after clipping
                if bbox[2] < 10 or bbox[3] < 10:
                    continue

                tile_annotations.append({
                    'caravan_id': caravan['id'],
                    'segmentation': clipped_coords,
                    'bbox': bbox
                })

            # Only keep tiles with enough caravans
            if len(tile_annotations) >= min_tile_caravans:
                # Calculate tile's geo bbox
                tile_west, tile_north = pixel_to_geo(x, y, img_bbox, full_image.size)
                tile_east, tile_south = pixel_to_geo(x + tile_size, y + tile_size, img_bbox, full_image.size)
                tile_bbox = (tile_west, tile_south, tile_east, tile_north)

                tiles.append({
                    'image': tile_img,
                    'annotations': tile_annotations,
                    'bbox': tile_bbox,
                    'park_id': park['id'],
                    'tile_idx': tile_idx
                })
            tile_idx += 1

            x += step
        y += step

    if verbose:
        total_annotations = sum(len(t['annotations']) for t in tiles)
        print(f"  Generated {len(tiles)} tiles ({tile_size}x{tile_size}), {total_annotations} total annotations")

    return tiles


def save_coco_dataset(all_tiles, output_dir, train_split=0.8, verbose=True):
    """Save tiles as COCO format dataset."""
    output_dir = Path(output_dir)

    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(all_tiles)
    split_idx = int(len(all_tiles) * train_split)
    train_tiles = all_tiles[:split_idx]
    val_tiles = all_tiles[split_idx:]

    def create_coco_json(tiles, split_name):
        images = []
        annotations = []
        ann_id = 1

        for img_id, tile in enumerate(tiles, 1):
            # Save image
            img_filename = f"park{tile['park_id']}_tile{tile['tile_idx']}.png"
            img_path = output_dir / split_name / "images" / img_filename
            tile['image'].save(img_path)

            images.append({
                "id": img_id,
                "file_name": img_filename,
                "width": tile['image'].size[0],
                "height": tile['image'].size[1]
            })

            for ann in tile['annotations']:
                # Flatten segmentation
                seg_flat = []
                for px, py in ann['segmentation']:
                    seg_flat.extend([px, py])

                # Calculate area
                xs = [p[0] for p in ann['segmentation']]
                ys = [p[1] for p in ann['segmentation']]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [seg_flat],
                    "bbox": ann['bbox'],
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

        return {
            "info": {
                "description": "Caravan Detection Dataset from OSM",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "categories": [{
                "id": 1,
                "name": "caravan",
                "supercategory": "building"
            }],
            "images": images,
            "annotations": annotations
        }

    # Create and save COCO JSONs
    train_coco = create_coco_json(train_tiles, "train")
    val_coco = create_coco_json(val_tiles, "val")

    with open(train_dir / "annotations.json", 'w') as f:
        json.dump(train_coco, f, indent=2)

    with open(val_dir / "annotations.json", 'w') as f:
        json.dump(val_coco, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("DATASET CREATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\nOutput: {output_dir}")
        print(f"\nTraining set:")
        print(f"  - Images: {len(train_coco['images'])}")
        print(f"  - Annotations: {len(train_coco['annotations'])}")
        print(f"\nValidation set:")
        print(f"  - Images: {len(val_coco['images'])}")
        print(f"  - Annotations: {len(val_coco['annotations'])}")
        print(f"\nFiles:")
        print(f"  {train_dir}/images/*.png")
        print(f"  {train_dir}/annotations.json")
        print(f"  {val_dir}/images/*.png")
        print(f"  {val_dir}/annotations.json")

    return train_coco, val_coco


def main():
    parser = argparse.ArgumentParser(
        description="Generate training dataset from fully-labeled OSM parks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from a single region
    python generate_training_from_osm.py --region anglesey --output training_data/

    # Generate from MULTIPLE regions
    python generate_training_from_osm.py --region anglesey --region cornwall --region norfolk --output training_data/

    # Generate from ALL available regions
    python generate_training_from_osm.py --all-regions --output training_data/

    # Match your inference setup (zoom 18, 2048x2048)
    python generate_training_from_osm.py --region cornwall --zoom 18 --tile-size 2048 --output data/

    # Download specific parks by OSM way ID
    python generate_training_from_osm.py --way-ids 123456789,987654321 --output data/
        """
    )

    parser.add_argument('--region', choices=UK_REGIONS.keys(), action='append',
                        help='Region(s) to search. Can specify multiple times: --region anglesey --region cornwall')
    parser.add_argument('--all-regions', action='store_true',
                        help='Search ALL available regions')
    parser.add_argument('--bbox', help='Custom bbox: "west,south,east,north"')
    parser.add_argument('--way-ids', help='Comma-separated OSM way IDs of specific parks')
    parser.add_argument('--output', default='training_data/', help='Output directory')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level (default: 18)')
    parser.add_argument('--tile-size', type=int, default=2048, help='Tile size in pixels (default: 2048)')
    parser.add_argument('--overlap', type=int, default=256, help='Tile overlap in pixels (default: 256)')
    parser.add_argument('--min-caravans', type=int, default=10, help='Minimum caravans per park')
    parser.add_argument('--max-parks-per-region', type=int, default=5, help='Maximum parks per region')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')

    args = parser.parse_args()

    # Determine regions to search
    if args.all_regions:
        regions = list(UK_REGIONS.keys())
    elif args.region:
        regions = args.region
    elif args.bbox or args.way_ids:
        regions = []  # Will use bbox or way_ids instead
    else:
        # Default to a few good regions
        regions = ['anglesey', 'cornwall', 'norfolk']
        print(f"No region specified, using defaults: {regions}")

    print("="*60)
    print("GENERATING TRAINING DATASET FROM OSM")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Zoom: {args.zoom}")
    print(f"  Tile size: {args.tile_size}x{args.tile_size}")
    print(f"  Overlap: {args.overlap}")
    print(f"  Output: {args.output}")

    all_tiles = []

    if args.way_ids:
        # Download specific parks
        way_ids = [int(w.strip()) for w in args.way_ids.split(',')]
        print(f"\nDownloading {len(way_ids)} specific parks...")

        for way_id in way_ids:
            data = query_osm(way_id=way_id)
            parks, caravans = parse_osm_data(data)

            if not parks:
                print(f"  Park {way_id} not found")
                continue

            parks = match_caravans_to_parks(parks, caravans)
            park = parks[0]

            if park['caravan_count'] < 3:
                print(f"  Park {way_id} has only {park['caravan_count']} caravans, skipping")
                continue

            tiles = create_tiles_from_park(park, args.zoom, args.tile_size, args.overlap)
            all_tiles.extend(tiles)
            time.sleep(2)

    else:
        # Search regions for parks
        if args.bbox:
            # Custom bbox
            regions_to_search = [('custom', args.bbox)]
        else:
            # Named regions
            regions_to_search = [(r, UK_REGIONS[r]) for r in regions]

        print(f"\nSearching {len(regions_to_search)} region(s): {[r[0] for r in regions_to_search]}")

        for region_name, bbox in regions_to_search:
            print(f"\n{'='*40}")
            print(f"Region: {region_name}")
            print(f"{'='*40}")

            try:
                data = query_osm(bbox=bbox)
                parks, caravans = parse_osm_data(data)
                parks = match_caravans_to_parks(parks, caravans)

                # Filter and sort
                parks = [p for p in parks if p['caravan_count'] >= args.min_caravans]
                parks.sort(key=lambda x: x['caravan_count'], reverse=True)
                parks = parks[:args.max_parks_per_region]

                print(f"Found {len(parks)} parks with {args.min_caravans}+ caravans")

                for park in parks:
                    tiles = create_tiles_from_park(park, args.zoom, args.tile_size, args.overlap)
                    all_tiles.extend(tiles)
                    time.sleep(2)

            except Exception as e:
                print(f"Error processing region {region_name}: {e}")
                continue

            time.sleep(3)  # Rate limit between regions

    if not all_tiles:
        print("\nNo tiles generated! Try a different region or lower --min-caravans")
        return 1

    # Filter to only tiles with caravans
    tiles_with_caravans = [t for t in all_tiles if len(t['annotations']) > 0]
    print(f"\nTotal tiles with caravans: {len(tiles_with_caravans)}")

    # Save dataset
    save_coco_dataset(tiles_with_caravans, args.output, args.train_split)

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nTo train with Detectron2, use this dataset path: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
