#!/usr/bin/env python3
"""
Caravan Dataset Creator

Creates a training dataset for caravan detection by:
1. Fetching caravan locations/polygons from OpenStreetMap
2. Checking OpenAerialMap for imagery coverage
3. Downloading aerial imagery tiles
4. Generating training patches and masks

Each step can be run independently for testing/inspection.

Usage:
    python create_caravan_dataset.py step1_osm --bbox "-4.7,53.2,-4.0,53.4" --output data/osm_caravans.geojson
    python create_caravan_dataset.py step2_coverage --input data/osm_caravans.geojson --output data/coverage.geojson
    python create_caravan_dataset.py step3_download --input data/coverage.geojson --output data/imagery/
    python create_caravan_dataset.py step4_generate --input data/imagery/ --output data/training/

    # Or run all steps at once:
    python create_caravan_dataset.py all --bbox "-4.7,53.2,-4.0,53.4" --output data/training/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import requests
import numpy as np
from PIL import Image, ImageDraw


# =============================================================================
# STEP 1: Fetch caravan data from OpenStreetMap
# =============================================================================

def step1_fetch_osm_caravans(bbox, output_path, verbose=True):
    """
    Fetch caravan polygons from OpenStreetMap using Overpass API.

    Args:
        bbox: Bounding box as "west,south,east,north" (lon,lat,lon,lat)
        output_path: Path to save GeoJSON output
        verbose: Print progress messages

    Returns:
        dict: GeoJSON FeatureCollection with caravan features
    """
    if verbose:
        print("=" * 60)
        print("STEP 1: Fetching caravan data from OpenStreetMap")
        print("=" * 60)

    # Parse bbox
    west, south, east, north = map(float, bbox.split(","))

    if verbose:
        print(f"Bounding box: {west}, {south}, {east}, {north}")

    # Overpass API query for caravan-related features
    # bbox format for Overpass is (south,west,north,east)
    overpass_query = f"""
    [out:json][timeout:120];
    (
        // Individual static caravans (polygons)
        way["building"="static_caravan"]({south},{west},{north},{east});

        // Caravan sites (areas with caravans)
        way["tourism"="caravan_site"]({south},{west},{north},{east});
        way["landuse"="static_caravan"]({south},{west},{north},{east});

        // Also get relations
        relation["building"="static_caravan"]({south},{west},{north},{east});
        relation["tourism"="caravan_site"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    if verbose:
        print("Querying Overpass API...")

    overpass_url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        return None

    if verbose:
        print(f"Received {len(data.get('elements', []))} elements from OSM")

    # Convert OSM data to GeoJSON
    geojson = osm_to_geojson(data, verbose)

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    if verbose:
        print(f"\nSaved {len(geojson['features'])} features to: {output_path}")
        print_feature_summary(geojson)

    return geojson


def osm_to_geojson(osm_data, verbose=True):
    """Convert OSM JSON to GeoJSON format."""

    # Build node lookup
    nodes = {}
    for element in osm_data.get('elements', []):
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])

    features = []

    for element in osm_data.get('elements', []):
        if element['type'] == 'way' and 'nodes' in element:
            # Get coordinates from node references
            coords = []
            for node_id in element['nodes']:
                if node_id in nodes:
                    coords.append(nodes[node_id])

            if len(coords) < 3:
                continue

            # Close polygon if not closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            # Determine feature type
            tags = element.get('tags', {})
            feature_type = None
            if tags.get('building') == 'static_caravan':
                feature_type = 'static_caravan'
            elif tags.get('tourism') == 'caravan_site':
                feature_type = 'caravan_site'
            elif tags.get('landuse') == 'static_caravan':
                feature_type = 'static_caravan_landuse'

            if feature_type:
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'osm_id': element['id'],
                        'feature_type': feature_type,
                        'tags': tags
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [coords]
                    }
                }
                features.append(feature)

    return {
        'type': 'FeatureCollection',
        'features': features
    }


def print_feature_summary(geojson):
    """Print summary of features by type."""
    counts = {}
    for feature in geojson['features']:
        ftype = feature['properties'].get('feature_type', 'unknown')
        counts[ftype] = counts.get(ftype, 0) + 1

    print("\nFeature summary:")
    for ftype, count in sorted(counts.items()):
        print(f"  - {ftype}: {count}")


# =============================================================================
# STEP 2: Check OpenAerialMap coverage
# =============================================================================

def step2_check_coverage(input_path, output_path, verbose=True):
    """
    Check which caravan locations have OpenAerialMap imagery coverage.

    Args:
        input_path: Path to GeoJSON from step 1
        output_path: Path to save coverage results
        verbose: Print progress messages

    Returns:
        dict: GeoJSON with coverage information added
    """
    if verbose:
        print("=" * 60)
        print("STEP 2: Checking OpenAerialMap coverage")
        print("=" * 60)

    # Load input
    with open(input_path, 'r') as f:
        geojson = json.load(f)

    if verbose:
        print(f"Loaded {len(geojson['features'])} features")

    # Get overall bounding box
    all_coords = []
    for feature in geojson['features']:
        coords = feature['geometry']['coordinates'][0]
        all_coords.extend(coords)

    if not all_coords:
        print("No coordinates found in input!")
        return None

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    if verbose:
        print(f"Overall bbox: {bbox}")
        print("Querying OpenAerialMap API...")

    # Query OAM API
    oam_url = "https://api.openaerialmap.org/meta"
    params = {
        'bbox': ','.join(map(str, bbox)),
        'limit': 500
    }

    try:
        response = requests.get(oam_url, params=params, timeout=60)
        response.raise_for_status()
        oam_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying OpenAerialMap API: {e}")
        # Continue without OAM data - we can use other sources
        oam_data = {'results': []}

    oam_images = oam_data.get('results', [])

    if verbose:
        print(f"Found {len(oam_images)} OpenAerialMap images in area")

    # Add coverage info to features
    coverage_info = {
        'oam_images': [],
        'alternative_sources': [
            {
                'name': 'ESRI World Imagery',
                'url_template': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'attribution': 'Esri, Maxar, Earthstar Geographics',
                'available': True
            },
            {
                'name': 'Bing Maps Aerial',
                'url_template': 'https://ecn.t{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1',
                'attribution': 'Microsoft Bing Maps',
                'note': 'Requires quadkey calculation',
                'available': True
            }
        ]
    }

    for img in oam_images:
        coverage_info['oam_images'].append({
            'id': img.get('_id'),
            'title': img.get('title'),
            'tms': img.get('tms'),
            'uuid': img.get('uuid'),
            'gsd': img.get('gsd'),  # Ground sample distance (resolution)
            'bbox': img.get('bbox'),
            'acquisition_start': img.get('acquisition_start'),
            'provider': img.get('provider')
        })

    # Create output with coverage info
    output_geojson = geojson.copy()
    output_geojson['coverage'] = coverage_info

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_geojson, f, indent=2)

    if verbose:
        print(f"\nSaved to: {output_path}")
        print(f"\nCoverage summary:")
        print(f"  - OpenAerialMap images: {len(oam_images)}")
        print(f"  - Alternative sources available: ESRI World Imagery, Bing Maps")

        if oam_images:
            print(f"\nOAM images found:")
            for img in oam_images[:5]:
                print(f"  - {img.get('title', 'Untitled')} (GSD: {img.get('gsd', 'N/A')}m)")
            if len(oam_images) > 5:
                print(f"  ... and {len(oam_images) - 5} more")

    return output_geojson


# =============================================================================
# STEP 3: Download imagery
# =============================================================================

def step3_download_imagery(input_path, output_dir, source='esri', zoom=18, verbose=True):
    """
    Download aerial imagery tiles for caravan locations.

    Args:
        input_path: Path to GeoJSON from step 2
        output_dir: Directory to save imagery
        source: Imagery source ('esri', 'oam', or 'bing')
        zoom: Zoom level (higher = more detail, 18-19 recommended)
        verbose: Print progress messages

    Returns:
        list: Paths to downloaded images with metadata
    """
    if verbose:
        print("=" * 60)
        print("STEP 3: Downloading aerial imagery")
        print("=" * 60)

    # Load input
    with open(input_path, 'r') as f:
        geojson = json.load(f)

    features = geojson['features']

    if verbose:
        print(f"Processing {len(features)} features")
        print(f"Source: {source}")
        print(f"Zoom level: {zoom}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for i, feature in enumerate(features):
        if verbose:
            print(f"\nProcessing feature {i+1}/{len(features)} (OSM ID: {feature['properties']['osm_id']})")

        # Get feature centroid
        coords = feature['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        # Calculate bounding box with padding
        padding = 0.0005  # ~50m padding
        feat_bbox = [
            min(lons) - padding,
            min(lats) - padding,
            max(lons) + padding,
            max(lats) + padding
        ]

        # Download imagery
        try:
            img_path = download_tile_image(
                center_lon, center_lat,
                output_dir / f"caravan_{feature['properties']['osm_id']}.png",
                source=source,
                zoom=zoom,
                size=512
            )

            if img_path:
                downloaded.append({
                    'image_path': str(img_path),
                    'feature': feature,
                    'center': [center_lon, center_lat],
                    'bbox': feat_bbox,
                    'zoom': zoom,
                    'source': source
                })

                if verbose:
                    print(f"  Downloaded: {img_path.name}")

        except Exception as e:
            print(f"  Error downloading: {e}")

        # Rate limiting
        time.sleep(0.5)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(downloaded, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Downloaded {len(downloaded)} images to: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")

    return downloaded


def download_tile_image(lon, lat, output_path, source='esri', zoom=18, size=512):
    """
    Download a tile image centered on the given coordinates.

    Args:
        lon, lat: Center coordinates
        output_path: Path to save the image
        source: 'esri' or 'oam'
        zoom: Zoom level
        size: Output image size in pixels

    Returns:
        Path to saved image or None if failed
    """
    # Convert lat/lon to tile coordinates
    n = 2 ** zoom
    tile_x = int((lon + 180) / 360 * n)
    tile_y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n)

    # Calculate how many tiles we need for the desired size
    tiles_needed = (size // 256) + 2  # 256 is standard tile size

    if source == 'esri':
        url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    else:
        # Default to ESRI
        url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    # Download center tile and surrounding tiles
    tiles = []
    half = tiles_needed // 2

    for dy in range(-half, half + 1):
        row = []
        for dx in range(-half, half + 1):
            tx, ty = tile_x + dx, tile_y + dy
            url = url_template.format(z=zoom, x=tx, y=ty)

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                tile_img = Image.open(requests.io.BytesIO(response.content))
                row.append(tile_img)
            except Exception as e:
                # Create blank tile on error
                row.append(Image.new('RGB', (256, 256), (128, 128, 128)))
        tiles.append(row)

    # Stitch tiles together
    total_width = len(tiles[0]) * 256
    total_height = len(tiles) * 256
    stitched = Image.new('RGB', (total_width, total_height))

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            stitched.paste(tile, (x * 256, y * 256))

    # Crop to center
    left = (total_width - size) // 2
    top = (total_height - size) // 2
    cropped = stitched.crop((left, top, left + size, top + size))

    # Save
    output_path = Path(output_path)
    cropped.save(output_path)

    return output_path


# Need to add this import for BytesIO
import io
requests.io = io


# =============================================================================
# STEP 4: Generate training data (patches + masks)
# =============================================================================

def step4_generate_training_data(input_dir, output_dir, patch_size=224, verbose=True):
    """
    Generate training patches and masks from downloaded imagery.

    Args:
        input_dir: Directory with downloaded imagery and metadata.json
        output_dir: Directory for training data output
        patch_size: Size of output patches
        verbose: Print progress messages

    Returns:
        dict: Statistics about generated data
    """
    if verbose:
        print("=" * 60)
        print("STEP 4: Generating training data")
        print("=" * 60)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Load metadata
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {input_dir}")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if verbose:
        print(f"Processing {len(metadata)} images")

    # Create output directories (matching expected structure)
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels" / "1"
    val_images = output_dir / "val" / "images"
    val_labels = output_dir / "val" / "labels" / "1"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'failed': 0}

    for i, item in enumerate(metadata):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(metadata)}")

        try:
            # Load image
            img_path = Path(item['image_path'])
            if not img_path.exists():
                stats['failed'] += 1
                continue

            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size

            # Create mask from polygon
            mask = create_mask_from_feature(
                item['feature'],
                item['bbox'],
                img_width,
                img_height
            )

            # Resize to patch size
            image_resized = image.resize((patch_size, patch_size), Image.LANCZOS)
            mask_resized = mask.resize((patch_size, patch_size), Image.NEAREST)

            # Split train/val (80/20)
            osm_id = item['feature']['properties']['osm_id']
            is_val = (i % 5 == 0)  # Every 5th image goes to validation

            if is_val:
                img_out = val_images / f"{osm_id}.tif"
                mask_out = val_labels / f"{osm_id}.png"
                stats['val'] += 1
            else:
                img_out = train_images / f"{osm_id}.tif"
                mask_out = train_labels / f"{osm_id}.png"
                stats['train'] += 1

            # Save
            image_resized.save(img_out)
            mask_resized.save(mask_out)

        except Exception as e:
            if verbose:
                print(f"Error processing {item.get('image_path', 'unknown')}: {e}")
            stats['failed'] += 1

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training data generated in: {output_dir}")
        print(f"\nStatistics:")
        print(f"  - Training images: {stats['train']}")
        print(f"  - Validation images: {stats['val']}")
        print(f"  - Failed: {stats['failed']}")
        print(f"\nDirectory structure:")
        print(f"  {output_dir}/")
        print(f"    train/")
        print(f"      images/    ({stats['train']} .tif files)")
        print(f"      labels/1/  ({stats['train']} .png masks)")
        print(f"    val/")
        print(f"      images/    ({stats['val']} .tif files)")
        print(f"      labels/1/  ({stats['val']} .png masks)")

    return stats


def create_mask_from_feature(feature, image_bbox, img_width, img_height):
    """
    Create a binary mask from a GeoJSON polygon feature.

    Args:
        feature: GeoJSON feature with polygon geometry
        image_bbox: [west, south, east, north] of the image
        img_width, img_height: Image dimensions in pixels

    Returns:
        PIL Image with mask (white = caravan, black = background)
    """
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    # Get polygon coordinates
    coords = feature['geometry']['coordinates'][0]

    # Convert geo coordinates to pixel coordinates
    west, south, east, north = image_bbox

    pixel_coords = []
    for lon, lat in coords:
        # Normalize to 0-1 range
        x_norm = (lon - west) / (east - west)
        y_norm = (north - lat) / (north - south)  # Flip Y axis

        # Convert to pixel coordinates
        px = int(x_norm * img_width)
        py = int(y_norm * img_height)
        pixel_coords.append((px, py))

    # Draw filled polygon
    if len(pixel_coords) >= 3:
        draw.polygon(pixel_coords, fill=255)

    return mask


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create caravan detection training dataset from OSM + aerial imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run each step individually:
  python create_caravan_dataset.py step1_osm --bbox "-4.7,53.2,-4.0,53.4" --output data/osm_caravans.geojson
  python create_caravan_dataset.py step2_coverage --input data/osm_caravans.geojson --output data/coverage.geojson
  python create_caravan_dataset.py step3_download --input data/coverage.geojson --output data/imagery/
  python create_caravan_dataset.py step4_generate --input data/imagery/ --output data/training/

  # Run all steps at once:
  python create_caravan_dataset.py all --bbox "-4.7,53.2,-4.0,53.4" --output data/training/

Bounding box format: "west,south,east,north" (longitude,latitude,longitude,latitude)

Example areas with caravans (UK):
  - Anglesey: "-4.7,53.1,-4.0,53.4"
  - Cornwall: "-5.1,50.0,-4.8,50.2"
  - Norfolk Broads: "1.4,52.6,1.7,52.8"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Step 1: OSM
    p1 = subparsers.add_parser('step1_osm', help='Fetch caravan data from OpenStreetMap')
    p1.add_argument('--bbox', required=True, help='Bounding box: "west,south,east,north"')
    p1.add_argument('--output', default='data/osm_caravans.geojson', help='Output GeoJSON path')

    # Step 2: Coverage
    p2 = subparsers.add_parser('step2_coverage', help='Check OpenAerialMap coverage')
    p2.add_argument('--input', required=True, help='Input GeoJSON from step 1')
    p2.add_argument('--output', default='data/coverage.geojson', help='Output GeoJSON path')

    # Step 3: Download
    p3 = subparsers.add_parser('step3_download', help='Download aerial imagery')
    p3.add_argument('--input', required=True, help='Input GeoJSON from step 2')
    p3.add_argument('--output', default='data/imagery/', help='Output directory')
    p3.add_argument('--source', default='esri', choices=['esri', 'oam'], help='Imagery source')
    p3.add_argument('--zoom', type=int, default=18, help='Zoom level (18-19 recommended)')

    # Step 4: Generate
    p4 = subparsers.add_parser('step4_generate', help='Generate training patches and masks')
    p4.add_argument('--input', required=True, help='Input directory from step 3')
    p4.add_argument('--output', default='data/training/', help='Output directory')
    p4.add_argument('--patch-size', type=int, default=224, help='Patch size in pixels')

    # All steps
    p_all = subparsers.add_parser('all', help='Run all steps')
    p_all.add_argument('--bbox', required=True, help='Bounding box: "west,south,east,north"')
    p_all.add_argument('--output', default='data/training/', help='Final output directory')
    p_all.add_argument('--source', default='esri', choices=['esri', 'oam'], help='Imagery source')
    p_all.add_argument('--zoom', type=int, default=18, help='Zoom level')
    p_all.add_argument('--patch-size', type=int, default=224, help='Patch size in pixels')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'step1_osm':
        result = step1_fetch_osm_caravans(args.bbox, args.output)
        return 0 if result else 1

    elif args.command == 'step2_coverage':
        result = step2_check_coverage(args.input, args.output)
        return 0 if result else 1

    elif args.command == 'step3_download':
        result = step3_download_imagery(args.input, args.output, args.source, args.zoom)
        return 0 if result else 1

    elif args.command == 'step4_generate':
        result = step4_generate_training_data(args.input, args.output, args.patch_size)
        return 0 if result else 1

    elif args.command == 'all':
        print("Running all steps...")
        print()

        # Temp paths
        base_dir = Path(args.output).parent
        osm_path = base_dir / "osm_caravans.geojson"
        coverage_path = base_dir / "coverage.geojson"
        imagery_dir = base_dir / "imagery"

        # Step 1
        result = step1_fetch_osm_caravans(args.bbox, osm_path)
        if not result or len(result['features']) == 0:
            print("\nNo caravans found in this area. Try a different bounding box.")
            return 1
        print()

        # Step 2
        result = step2_check_coverage(osm_path, coverage_path)
        if not result:
            return 1
        print()

        # Step 3
        result = step3_download_imagery(coverage_path, imagery_dir, args.source, args.zoom)
        if not result:
            return 1
        print()

        # Step 4
        result = step4_generate_training_data(imagery_dir, args.output, args.patch_size)
        if not result:
            return 1

        print()
        print("=" * 60)
        print("ALL STEPS COMPLETE!")
        print("=" * 60)
        print(f"\nTraining data ready in: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review the generated images and masks")
        print(f"  2. Train the model:")
        print(f"     python samples/caravans/caravan.py train --dataset={args.output} --weights=coco")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
