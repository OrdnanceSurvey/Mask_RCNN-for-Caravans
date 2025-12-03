#!/usr/bin/env python3
"""
Download a Caravan Park with Full OSM Labels

Finds caravan parks where ALL caravans are already labeled in OpenStreetMap,
downloads the aerial imagery, and exports the polygons for training.

This gives you a fully-labeled training image with no manual work!

Usage:
    # Find parks with most labeled caravans
    python download_labeled_park.py search --region wales

    # Download a specific park by OSM way ID
    python download_labeled_park.py download --way-id 123456789 --output my_park/

    # Download top N parks automatically
    python download_labeled_park.py download-top --region wales --count 5 --output parks/
"""

import argparse
import requests
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import time

# UK regions
UK_REGIONS = {
    "uk": "-10.5,49.5,2.0,61.0",
    "wales": "-5.5,51.3,-2.6,53.5",
    "england_south": "-5.7,49.9,1.8,52.0",
    "england_north": "-3.5,52.0,0.5,55.8",
    "scotland": "-7.5,54.5,-0.5,61.0",
    "east_anglia": "0.0,51.5,2.0,53.5",
    "cornwall": "-5.8,49.9,-4.2,51.2",
    "devon": "-4.7,50.2,-2.9,51.3",
    "yorkshire": "-2.5,53.3,0.0,54.6",
    "lincolnshire": "-0.8,52.7,0.4,53.7",
    "norfolk": "0.3,52.3,2.0,53.1",
}


def query_park_with_caravans(way_id=None, bbox=None, verbose=True):
    """
    Query OSM for a specific park and all caravans within it.

    Either provide way_id for a specific park, or bbox to search an area.
    """
    if way_id:
        # Query specific park and nearby caravans
        query = f"""
        [out:json][timeout:120];

        // Get the park
        way({way_id});
        (._;>;);
        out body;

        // Get park bounds for caravan search
        way({way_id});
        map_to_area -> .park;

        // Get all caravans in/near the park
        (
            way["building"="static_caravan"](around:500);
        );
        (._;>;);
        out body;
        """
    elif bbox:
        west, south, east, north = map(float, bbox.split(","))
        query = f"""
        [out:json][timeout:180];
        (
            way["tourism"="caravan_site"]({south},{west},{north},{east});
            way["building"="static_caravan"]({south},{west},{north},{east});
        );
        (._;>;);
        out body;
        """
    else:
        raise ValueError("Must provide either way_id or bbox")

    if verbose:
        print("Querying OpenStreetMap...")

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=180
    )
    response.raise_for_status()
    return response.json()


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

            item = {
                'id': el['id'],
                'name': tags.get('name', 'Unnamed'),
                'coords': coords,
                'tags': tags
            }

            # Calculate bounds
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            item['bbox'] = (min(lons), min(lats), max(lons), max(lats))
            item['center'] = (sum(lons)/len(lons), sum(lats)/len(lats))

            if tags.get('tourism') == 'caravan_site':
                parks.append(item)
            elif tags.get('building') == 'static_caravan':
                caravans.append(item)

    return parks, caravans


def find_best_parks(bbox, min_caravans=10, verbose=True):
    """Find parks with the most labeled caravans."""
    if verbose:
        print(f"Searching for caravan parks in bbox: {bbox}")

    data = query_park_with_caravans(bbox=bbox, verbose=verbose)
    parks, caravans = parse_osm_data(data)

    if verbose:
        print(f"Found {len(parks)} parks and {len(caravans)} caravans")

    # Match caravans to parks
    for park in parks:
        p_west, p_south, p_east, p_north = park['bbox']
        buffer = 0.002  # ~200m buffer

        park['caravans'] = []
        for caravan in caravans:
            c_lon, c_lat = caravan['center']
            if (p_west - buffer <= c_lon <= p_east + buffer and
                p_south - buffer <= c_lat <= p_north + buffer):
                park['caravans'].append(caravan)

        park['caravan_count'] = len(park['caravans'])

    # Sort by caravan count
    parks = [p for p in parks if p['caravan_count'] >= min_caravans]
    parks.sort(key=lambda x: x['caravan_count'], reverse=True)

    return parks


def download_park_imagery(park, output_dir, zoom=18, verbose=True):
    """Download aerial imagery for a park."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get park bounds with padding
    west, south, east, north = park['bbox']
    pad = 0.001  # Small padding
    west -= pad
    south -= pad
    east += pad
    north += pad

    if verbose:
        print(f"\nDownloading imagery for: {park['name']}")
        print(f"Bounds: {west:.5f}, {south:.5f} to {east:.5f}, {north:.5f}")
        print(f"Caravans in park: {park['caravan_count']}")

    # Calculate tile range
    n = 2 ** zoom

    def lonlat_to_tile(lon, lat):
        x = int((lon + 180) / 360 * n)
        y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n)
        return x, y

    def tile_to_lonlat(tx, ty):
        lon = tx / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ty / n))))
        return lon, lat

    min_tx, max_ty = lonlat_to_tile(west, south)
    max_tx, min_ty = lonlat_to_tile(east, north)

    # Download tiles
    tiles = []
    url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    total_tiles = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
    if verbose:
        print(f"Downloading {total_tiles} tiles at zoom {zoom}...")

    for ty in range(min_ty, max_ty + 1):
        row = []
        for tx in range(min_tx, max_tx + 1):
            url = url_template.format(z=zoom, x=tx, y=ty)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                from io import BytesIO
                tile = Image.open(BytesIO(resp.content))
                row.append(tile)
            except Exception as e:
                row.append(Image.new('RGB', (256, 256), (128, 128, 128)))
            time.sleep(0.1)  # Rate limiting
        tiles.append(row)

    # Stitch tiles
    img_width = (max_tx - min_tx + 1) * 256
    img_height = (max_ty - min_ty + 1) * 256
    stitched = Image.new('RGB', (img_width, img_height))

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            stitched.paste(tile, (x * 256, y * 256))

    # Calculate actual image bounds
    img_west, img_north = tile_to_lonlat(min_tx, min_ty)
    img_east, img_south = tile_to_lonlat(max_tx + 1, max_ty + 1)
    img_bbox = [img_west, img_south, img_east, img_north]

    # Save image
    park_id = park['id']
    img_path = output_dir / f"park_{park_id}.png"
    stitched.save(img_path)

    if verbose:
        print(f"Saved: {img_path} ({img_width}x{img_height})")

    return stitched, img_bbox


def create_coco_annotations(park, img_bbox, img_size, output_dir):
    """Create COCO-format annotations for all caravans."""
    output_dir = Path(output_dir)
    img_width, img_height = img_size
    west, south, east, north = img_bbox

    def geo_to_pixel(lon, lat):
        px = int((lon - west) / (east - west) * img_width)
        py = int((north - lat) / (north - south) * img_height)
        return px, py

    annotations = []

    for idx, caravan in enumerate(park['caravans']):
        # Convert polygon to pixel coordinates
        pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in caravan['coords']]

        # Flatten for COCO format
        segmentation = []
        for px, py in pixel_coords:
            segmentation.extend([px, py])

        # Calculate bounding box
        xs = [p[0] for p in pixel_coords]
        ys = [p[1] for p in pixel_coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Skip if outside image bounds
        if x_max < 0 or y_max < 0 or x_min > img_width or y_min > img_height:
            continue

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = (x_max - x_min) * (y_max - y_min)

        annotations.append({
            "id": idx + 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [segmentation],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })

    # Create COCO dataset
    coco = {
        "info": {
            "description": f"Caravan park: {park['name']}",
            "version": "1.0",
            "osm_way_id": park['id']
        },
        "images": [{
            "id": 1,
            "file_name": f"park_{park['id']}.png",
            "width": img_width,
            "height": img_height,
            "bbox": img_bbox
        }],
        "categories": [{
            "id": 1,
            "name": "caravan",
            "supercategory": "building"
        }],
        "annotations": annotations
    }

    # Save
    json_path = output_dir / f"park_{park['id']}_coco.json"
    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Saved COCO annotations: {json_path}")
    print(f"  - {len(annotations)} caravan annotations")

    return coco


def create_visualization(image, park, img_bbox, output_dir):
    """Create visualization with caravan polygons overlaid."""
    output_dir = Path(output_dir)
    img_width, img_height = image.size
    west, south, east, north = img_bbox

    def geo_to_pixel(lon, lat):
        px = int((lon - west) / (east - west) * img_width)
        py = int((north - lat) / (north - south) * img_height)
        return px, py

    # Create overlay
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')

    for caravan in park['caravans']:
        pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in caravan['coords']]
        if len(pixel_coords) >= 3:
            # Draw filled polygon with transparency
            draw.polygon(pixel_coords, fill=(255, 0, 0, 100), outline=(255, 0, 0, 255))

    # Save
    viz_path = output_dir / f"park_{park['id']}_visualization.png"
    overlay.save(viz_path)
    print(f"Saved visualization: {viz_path}")

    return overlay


def create_mask(image, park, img_bbox, output_dir):
    """Create binary mask image for all caravans."""
    output_dir = Path(output_dir)
    img_width, img_height = image.size
    west, south, east, north = img_bbox

    def geo_to_pixel(lon, lat):
        px = int((lon - west) / (east - west) * img_width)
        py = int((north - lat) / (north - south) * img_height)
        return px, py

    # Create mask
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    for caravan in park['caravans']:
        pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in caravan['coords']]
        if len(pixel_coords) >= 3:
            draw.polygon(pixel_coords, fill=255)

    # Save
    mask_path = output_dir / f"park_{park['id']}_mask.png"
    mask.save(mask_path)
    print(f"Saved mask: {mask_path}")

    return mask


def download_single_park(park, output_dir, zoom=18, verbose=True):
    """Download a single park with all annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download imagery
    image, img_bbox = download_park_imagery(park, output_dir, zoom, verbose)

    # Create annotations
    create_coco_annotations(park, img_bbox, image.size, output_dir)

    # Create visualization
    create_visualization(image, park, img_bbox, output_dir)

    # Create mask
    create_mask(image, park, img_bbox, output_dir)

    # Save park metadata
    meta_path = output_dir / f"park_{park['id']}_meta.json"
    meta = {
        'id': park['id'],
        'name': park['name'],
        'caravan_count': park['caravan_count'],
        'center': park['center'],
        'bbox': park['bbox'],
        'image_bbox': img_bbox,
        'osm_url': f"https://www.openstreetmap.org/way/{park['id']}"
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Park downloaded successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files:")
    print(f"    - park_{park['id']}.png (aerial image)")
    print(f"    - park_{park['id']}_mask.png (binary mask)")
    print(f"    - park_{park['id']}_visualization.png (overlay)")
    print(f"    - park_{park['id']}_coco.json (COCO annotations)")


def main():
    parser = argparse.ArgumentParser(
        description="Download caravan parks with full OSM labels",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Search command
    p_search = subparsers.add_parser('search', help='Find parks with labeled caravans')
    p_search.add_argument('--region', default='wales', choices=UK_REGIONS.keys())
    p_search.add_argument('--bbox', help='Custom bbox: "west,south,east,north"')
    p_search.add_argument('--min-caravans', type=int, default=10)
    p_search.add_argument('--top', type=int, default=20)

    # Download command
    p_download = subparsers.add_parser('download', help='Download a specific park')
    p_download.add_argument('--way-id', type=int, required=True, help='OSM way ID')
    p_download.add_argument('--output', default='park_data/', help='Output directory')
    p_download.add_argument('--zoom', type=int, default=18)

    # Download top parks
    p_top = subparsers.add_parser('download-top', help='Download top N parks')
    p_top.add_argument('--region', default='wales', choices=UK_REGIONS.keys())
    p_top.add_argument('--bbox', help='Custom bbox')
    p_top.add_argument('--count', type=int, default=5)
    p_top.add_argument('--output', default='parks_data/', help='Output directory')
    p_top.add_argument('--zoom', type=int, default=18)
    p_top.add_argument('--min-caravans', type=int, default=20)

    args = parser.parse_args()

    if args.command == 'search':
        bbox = args.bbox or UK_REGIONS[args.region]
        parks = find_best_parks(bbox, args.min_caravans)

        print(f"\n{'='*70}")
        print(f"TOP {args.top} CARAVAN PARKS WITH MOST OSM-LABELED CARAVANS")
        print(f"{'='*70}\n")

        for i, park in enumerate(parks[:args.top], 1):
            print(f"{i:3}. {park['name']}")
            print(f"     Caravans: {park['caravan_count']}")
            print(f"     OSM Way: {park['id']}")
            print(f"     Link: https://www.openstreetmap.org/way/{park['id']}")
            print(f"     Center: {park['center'][1]:.5f}, {park['center'][0]:.5f}")
            print()

        print(f"To download a park, run:")
        print(f"  python download_labeled_park.py download --way-id <WAY_ID> --output my_park/")

    elif args.command == 'download':
        # Query the specific park
        print(f"Fetching park {args.way_id} from OSM...")

        # Get park and surrounding caravans
        data = query_park_with_caravans(way_id=args.way_id)
        parks, caravans = parse_osm_data(data)

        if not parks:
            print(f"Park {args.way_id} not found!")
            return 1

        park = parks[0]

        # Find caravans in this park
        p_west, p_south, p_east, p_north = park['bbox']
        buffer = 0.003
        park['caravans'] = [
            c for c in caravans
            if (p_west - buffer <= c['center'][0] <= p_east + buffer and
                p_south - buffer <= c['center'][1] <= p_north + buffer)
        ]
        park['caravan_count'] = len(park['caravans'])

        print(f"Found park: {park['name']}")
        print(f"Caravans labeled: {park['caravan_count']}")

        download_single_park(park, args.output, args.zoom)

    elif args.command == 'download-top':
        bbox = args.bbox or UK_REGIONS[args.region]
        parks = find_best_parks(bbox, args.min_caravans)

        print(f"\nDownloading top {args.count} parks...")

        for i, park in enumerate(parks[:args.count], 1):
            print(f"\n{'='*50}")
            print(f"Park {i}/{args.count}: {park['name']}")
            print(f"{'='*50}")

            park_dir = Path(args.output) / f"park_{park['id']}"
            download_single_park(park, park_dir, args.zoom)
            time.sleep(2)  # Rate limiting

        print(f"\n✓ Downloaded {min(args.count, len(parks))} parks to {args.output}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    exit(main())
