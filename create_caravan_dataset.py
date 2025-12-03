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

def step3_download_imagery(input_path, output_dir, source='esri', zoom=19, size=256, tight_crop=True, padding=1.0, verbose=True):
    """
    Download aerial imagery tiles for caravan locations.

    Args:
        input_path: Path to GeoJSON from step 2
        output_dir: Directory to save imagery
        source: Imagery source ('esri', 'oam', or 'bing')
        zoom: Zoom level (19 recommended for tight caravan crops)
        size: Image size in pixels (used if tight_crop=False)
        tight_crop: If True, crop based on caravan polygon size + padding
        padding: Padding multiplier around caravan (1.0 = 100% padding = 2x caravan size)
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

        # Calculate feature size in degrees
        feat_width_deg = max(lons) - min(lons)
        feat_height_deg = max(lats) - min(lats)

        # Calculate crop size based on caravan size
        if tight_crop:
            # Convert degrees to approximate pixels at this zoom level
            # At zoom z, the world is 256 * 2^z pixels wide
            # 360 degrees = 256 * 2^z pixels
            pixels_per_degree_lon = (256 * (2 ** zoom)) / 360
            # Latitude is more complex due to Mercator projection, approximate for now
            pixels_per_degree_lat = pixels_per_degree_lon * np.cos(np.radians(center_lat))

            feat_width_px = feat_width_deg * pixels_per_degree_lon
            feat_height_px = feat_height_deg * pixels_per_degree_lat

            # Make it square, use the larger dimension
            feat_size_px = max(feat_width_px, feat_height_px)

            # Add padding (e.g., padding=1.0 means 100% padding on each side = 3x total size)
            crop_size = int(feat_size_px * (1 + 2 * padding))

            # Ensure minimum size of 64 and maximum of 512
            crop_size = max(64, min(512, crop_size))

            # Round to nearest multiple of 32 for efficiency
            crop_size = ((crop_size + 31) // 32) * 32

            if verbose and i < 5:  # Show size calculation for first 5
                print(f"  Caravan size: {feat_width_px:.0f}x{feat_height_px:.0f}px -> crop: {crop_size}x{crop_size}px")
        else:
            crop_size = size

        # Feature bounding box (for reference)
        feat_bbox = [min(lons), min(lats), max(lons), max(lats)]

        # Download imagery
        try:
            result = download_tile_image(
                center_lon, center_lat,
                output_dir / f"caravan_{feature['properties']['osm_id']}.png",
                source=source,
                zoom=zoom,
                size=crop_size
            )

            if result and result[0]:
                img_path, actual_bbox = result
                downloaded.append({
                    'image_path': str(img_path),
                    'feature': feature,
                    'center': [center_lon, center_lat],
                    'bbox': actual_bbox,  # Use the ACTUAL image bbox, not feature bbox
                    'feature_bbox': feat_bbox,  # Keep feature bbox for reference
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


def tile_to_lonlat(tx, ty, zoom):
    """Convert tile coordinates to lon/lat (top-left corner of tile)."""
    n = 2 ** zoom
    lon_deg = tx / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ty / n)))
    lat_deg = np.degrees(lat_rad)
    return lon_deg, lat_deg


def download_tile_image(lon, lat, output_path, source='esri', zoom=18, size=256):
    """
    Download a tile image centered on the given coordinates.

    Args:
        lon, lat: Center coordinates
        output_path: Path to save the image
        source: 'esri' or 'oam'
        zoom: Zoom level (18-19 recommended for caravans)
        size: Output image size in pixels (256 recommended for tight crops)

    Returns:
        tuple: (Path to saved image, actual_bbox) where actual_bbox is [west, south, east, north]
               Returns (None, None) if failed
    """
    # Convert lat/lon to FRACTIONAL tile coordinates
    n = 2 ** zoom
    tile_x_frac = (lon + 180) / 360 * n
    tile_y_frac = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n

    # Integer tile coordinates
    tile_x = int(tile_x_frac)
    tile_y = int(tile_y_frac)

    # Fractional position within the tile (0-1)
    frac_x = tile_x_frac - tile_x
    frac_y = tile_y_frac - tile_y

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

    # Track the actual tile range
    min_tx = tile_x - half
    max_tx = tile_x + half
    min_ty = tile_y - half
    max_ty = tile_y + half

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

    # Calculate the EXACT pixel position of the center point in the stitched image
    # The center tile (tile_x, tile_y) is at grid position (half, half)
    # Within that tile, the center point is at fractional position (frac_x, frac_y)
    center_px_x = half * 256 + int(frac_x * 256)
    center_px_y = half * 256 + int(frac_y * 256)

    # Crop centered on the actual center point
    left = center_px_x - size // 2
    top = center_px_y - size // 2

    # Ensure we don't go out of bounds
    left = max(0, min(left, total_width - size))
    top = max(0, min(top, total_height - size))

    cropped = stitched.crop((left, top, left + size, top + size))

    # Calculate the ACTUAL bounding box of the cropped image
    # First, get the bbox of the full stitched image (in lon/lat)
    stitch_west, stitch_north = tile_to_lonlat(min_tx, min_ty, zoom)
    stitch_east, stitch_south = tile_to_lonlat(max_tx + 1, max_ty + 1, zoom)

    # Calculate the crop offset as a fraction of the stitched image
    crop_left_frac = left / total_width
    crop_top_frac = top / total_height
    crop_right_frac = (left + size) / total_width
    crop_bottom_frac = (top + size) / total_height

    # Apply the crop to get the actual image bbox
    stitch_lon_range = stitch_east - stitch_west
    stitch_lat_range = stitch_north - stitch_south  # Note: north > south

    actual_west = stitch_west + crop_left_frac * stitch_lon_range
    actual_east = stitch_west + crop_right_frac * stitch_lon_range
    actual_north = stitch_north - crop_top_frac * stitch_lat_range
    actual_south = stitch_north - crop_bottom_frac * stitch_lat_range

    actual_bbox = [actual_west, actual_south, actual_east, actual_north]

    # Save
    output_path = Path(output_path)
    cropped.save(output_path)

    return output_path, actual_bbox


# Need to add this import for BytesIO
import io
requests.io = io


# =============================================================================
# Empty plot detection
# =============================================================================

def is_empty_plot(image, mask, threshold=0.6):
    """
    Detect if a caravan plot is empty (just grass) by analyzing the masked region.

    Empty plots have:
    - Low color variance (uniformly green)
    - Low edge density (no sharp edges)
    - Similar appearance to surrounding grass

    Args:
        image: PIL Image (RGB)
        mask: PIL Image (L) with polygon mask
        threshold: Score threshold (0-1). Higher = stricter filtering

    Returns:
        bool: True if the plot appears empty (no caravan), False if occupied
    """
    img_array = np.array(image)
    mask_array = np.array(mask)

    # Get pixels inside the mask
    mask_bool = mask_array > 0
    if mask_bool.sum() < 50:  # Too few pixels to analyze
        return True

    masked_pixels = img_array[mask_bool]

    # 1. Color variance analysis
    # Caravans tend to be white/cream/tan, grass is green
    # Calculate standard deviation of each color channel
    r_std = np.std(masked_pixels[:, 0])
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    color_variance = (r_std + g_std + b_std) / 3

    # 2. Check if predominantly green (grass)
    r_mean = np.mean(masked_pixels[:, 0])
    g_mean = np.mean(masked_pixels[:, 1])
    b_mean = np.mean(masked_pixels[:, 2])

    # Grass has higher green relative to red and blue
    green_ratio = g_mean / (r_mean + b_mean + 1)
    is_green = green_ratio > 0.55  # Mostly green

    # 3. Brightness analysis
    # Caravans are usually brighter (white/cream colored)
    brightness = (r_mean + g_mean + b_mean) / 3
    is_dark = brightness < 100  # Darker pixels suggest shadow/grass

    # 4. Edge detection within the masked region
    # Convert to grayscale and detect edges
    gray = np.mean(img_array, axis=2)

    # Simple edge detection using gradient
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    # Pad to match original size
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')

    edges = np.sqrt(dx**2 + dy**2)
    edge_mean = np.mean(edges[mask_bool])

    # 5. Contrast with surrounding area
    # Dilate mask to get surrounding pixels
    try:
        from scipy import ndimage
        dilated = ndimage.binary_dilation(mask_bool, iterations=5)
        surrounding = dilated & ~mask_bool

        if surrounding.sum() > 10:
            surrounding_brightness = np.mean(img_array[surrounding])
            masked_brightness = np.mean(masked_pixels)
            contrast = abs(masked_brightness - surrounding_brightness)
        else:
            contrast = 0
    except ImportError:
        # scipy not available, skip contrast check
        contrast = 0

    # Scoring: higher score = more likely to be a caravan
    score = 0

    # Color variance (caravans have varied colors from roof, walls, etc.)
    if color_variance > 20:
        score += 0.3
    elif color_variance > 10:
        score += 0.15

    # Not predominantly green
    if not is_green:
        score += 0.25

    # Brighter than typical grass
    if brightness > 120:
        score += 0.2
    elif brightness > 100:
        score += 0.1

    # Has edges (rectangular structure)
    if edge_mean > 15:
        score += 0.25
    elif edge_mean > 8:
        score += 0.1

    # Contrasts with surroundings
    if contrast > 30:
        score += 0.15
    elif contrast > 15:
        score += 0.08

    # Return True (empty) if score is below threshold
    return score < threshold


# =============================================================================
# STEP 4: Generate training data (patches + masks)
# =============================================================================

def step4_generate_training_data(input_dir, output_dir, patch_size=None, filter_empty=True, verbose=True):
    """
    Generate training patches and masks from downloaded imagery.

    Args:
        input_dir: Directory with downloaded imagery and metadata.json
        output_dir: Directory for training data output
        patch_size: Size of output patches (None = keep original size)
        filter_empty: If True, skip images where caravan plot appears empty
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

    stats = {'train': 0, 'val': 0, 'failed': 0, 'empty': 0}

    if verbose and filter_empty:
        print("Empty plot filtering: ENABLED")

    for i, item in enumerate(metadata):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(metadata)} (train: {stats['train']}, val: {stats['val']}, empty: {stats['empty']})")

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

            # Filter out empty plots (just grass, no caravan)
            if filter_empty:
                try:
                    if is_empty_plot(image, mask):
                        stats['empty'] += 1
                        continue
                except Exception as e:
                    # If filtering fails, include the image anyway
                    pass

            # Resize to patch size if specified, otherwise keep original
            if patch_size:
                image_resized = image.resize((patch_size, patch_size), Image.LANCZOS)
                mask_resized = mask.resize((patch_size, patch_size), Image.NEAREST)
            else:
                image_resized = image
                mask_resized = mask

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
        print(f"  - Empty plots filtered: {stats['empty']}")
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
# Verification / Debug utilities
# =============================================================================

def verify_mask_alignment(training_dir, num_samples=10, output_path=None):
    """
    Verify that masks are properly aligned with images.
    Creates overlay visualizations to check alignment.

    Args:
        training_dir: Path to training data directory
        num_samples: Number of samples to visualize
        output_path: Optional path to save the visualization

    Returns:
        None (displays or saves visualization)
    """
    import matplotlib.pyplot as plt

    training_dir = Path(training_dir)
    train_images = training_dir / "train" / "images"
    train_labels = training_dir / "train" / "labels" / "1"

    # Get image files
    image_files = list(train_images.glob("*.tif")) + list(train_images.glob("*.png"))

    if not image_files:
        print(f"No images found in {train_images}")
        return

    # Sample random images
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    # Create visualization
    cols = min(5, len(samples))
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if len(samples) == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, img_path in enumerate(samples):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Find corresponding mask
        mask_name = img_path.stem + ".png"
        mask_path = train_labels / mask_name

        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')

            # Create overlay: red where mask is white
            img_array = np.array(img)
            mask_array = np.array(mask)

            # Create red overlay
            overlay = img_array.copy()
            mask_bool = mask_array > 0
            overlay[mask_bool, 0] = np.minimum(255, overlay[mask_bool, 0] + 100)  # Add red
            overlay[mask_bool, 1] = overlay[mask_bool, 1] // 2  # Reduce green
            overlay[mask_bool, 2] = overlay[mask_bool, 2] // 2  # Reduce blue

            ax.imshow(overlay)
            ax.set_title(f"{img_path.stem}\nmask pixels: {mask_bool.sum()}")
        else:
            ax.imshow(img)
            ax.set_title(f"{img_path.stem}\nNO MASK FOUND")

        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(samples), rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.suptitle("Mask Alignment Check (red = mask overlay)", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Main CLI
# =============================================================================

# UK regions known to have many static caravans
UK_CARAVAN_REGIONS = [
    {"name": "Anglesey", "bbox": "-4.7,53.1,-4.0,53.5"},
    {"name": "North Wales Coast", "bbox": "-4.0,53.1,-3.3,53.4"},
    {"name": "Cornwall", "bbox": "-5.7,49.9,-4.5,50.7"},
    {"name": "Devon South", "bbox": "-4.2,50.2,-3.4,50.7"},
    {"name": "Dorset", "bbox": "-2.9,50.5,-1.8,50.9"},
    {"name": "Norfolk", "bbox": "0.5,52.5,1.8,53.0"},
    {"name": "Suffolk Coast", "bbox": "1.2,51.9,1.8,52.5"},
    {"name": "Essex Coast", "bbox": "0.5,51.5,1.2,51.9"},
    {"name": "Kent Coast", "bbox": "0.8,50.9,1.4,51.4"},
    {"name": "East Yorkshire", "bbox": "-0.5,53.6,0.2,54.2"},
    {"name": "Lincolnshire Coast", "bbox": "-0.2,53.0,0.4,53.6"},
    {"name": "Cumbria Coast", "bbox": "-3.6,54.0,-3.0,54.5"},
    {"name": "Lancashire Coast", "bbox": "-3.2,53.5,-2.8,54.0"},
    {"name": "Scottish Borders", "bbox": "-2.5,55.4,-1.8,55.9"},
    {"name": "Pembrokeshire", "bbox": "-5.3,51.6,-4.7,52.0"},
]


def step_large_dataset(output_dir, target_count=1000, source='esri', zoom=19, tight_crop=True, padding=0.5, size=256, filter_empty=True, verbose=True):
    """
    Create a large dataset by fetching caravans from multiple UK regions.

    Args:
        output_dir: Directory for final training data
        target_count: Target number of caravan images (default: 1000)
        source: Imagery source
        zoom: Zoom level (19 recommended for tight crops)
        tight_crop: If True, crop based on caravan size + padding
        padding: Padding multiplier (0.5 = 50% padding on each side)
        size: Fixed image size (only used if tight_crop=False)
        filter_empty: If True, filter out empty caravan plots
        verbose: Print progress

    Returns:
        Statistics about the generated dataset
    """
    if verbose:
        print("=" * 60)
        print(f"LARGE DATASET: Targeting {target_count} caravan images")
        print("=" * 60)
        print(f"\nWill search {len(UK_CARAVAN_REGIONS)} UK regions for caravans")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    total_downloaded = 0

    # Step 1: Collect caravans from all regions
    for region in UK_CARAVAN_REGIONS:
        if total_downloaded >= target_count:
            break

        if verbose:
            print(f"\n{'='*40}")
            print(f"Searching: {region['name']}")
            print(f"{'='*40}")

        # Fetch OSM data for this region
        temp_osm = output_dir / f"temp_osm_{region['name'].replace(' ', '_')}.geojson"
        result = step1_fetch_osm_caravans(region['bbox'], temp_osm, verbose=verbose)

        if result and result.get('features'):
            features = result['features']
            # Only take static_caravan features (individual caravans, not sites)
            caravan_features = [f for f in features if f['properties'].get('feature_type') == 'static_caravan']

            if verbose:
                print(f"Found {len(caravan_features)} individual caravans in {region['name']}")

            all_features.extend(caravan_features)
            total_downloaded = len(all_features)

            if verbose:
                print(f"Total caravans so far: {total_downloaded}")

        time.sleep(1)  # Rate limiting between regions

    if verbose:
        print(f"\n{'='*60}")
        print(f"Collected {len(all_features)} caravan polygons from OSM")
        print(f"{'='*60}")

    if len(all_features) == 0:
        print("No caravans found! Check your internet connection.")
        return None

    # Limit to target count
    if len(all_features) > target_count:
        all_features = all_features[:target_count]
        if verbose:
            print(f"Limited to {target_count} caravans")

    # Save combined GeoJSON
    combined_geojson = {
        'type': 'FeatureCollection',
        'features': all_features
    }
    combined_path = output_dir / "all_caravans.geojson"
    with open(combined_path, 'w') as f:
        json.dump(combined_geojson, f)

    if verbose:
        print(f"Saved combined data to: {combined_path}")

    # Step 2: Check coverage (quick pass)
    coverage_path = output_dir / "coverage.geojson"
    step2_check_coverage(combined_path, coverage_path, verbose=verbose)

    # Step 3: Download imagery (tight crops around each caravan)
    imagery_dir = output_dir / "imagery"
    step3_download_imagery(coverage_path, imagery_dir, source=source, zoom=zoom,
                          size=size, tight_crop=tight_crop, padding=padding, verbose=verbose)

    # Step 4: Generate training data
    training_dir = output_dir / "training"
    stats = step4_generate_training_data(imagery_dir, training_dir, filter_empty=filter_empty, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print("LARGE DATASET COMPLETE!")
        print(f"{'='*60}")
        print(f"\nTraining data ready in: {training_dir}")
        print(f"\nTo train the model, run:")
        print(f"  python samples/caravans/caravan.py train --dataset={training_dir} --weights=coco")

    return stats


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

  # Create a large dataset (1000+ images) from multiple UK regions:
  python create_caravan_dataset.py large --count 1000 --output data/large_dataset/

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
    p3.add_argument('--zoom', type=int, default=19, help='Zoom level (19 recommended for tight crops)')
    p3.add_argument('--size', type=int, default=256, help='Image size in pixels (256 for single-caravan crops)')

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

    # Large dataset from multiple UK regions
    p_large = subparsers.add_parser('large', help='Create large dataset (1000+ images) from multiple UK regions')
    p_large.add_argument('--count', type=int, default=1000, help='Target number of images (default: 1000)')
    p_large.add_argument('--output', default='data/large_dataset/', help='Output directory')
    p_large.add_argument('--source', default='esri', choices=['esri', 'oam'], help='Imagery source')
    p_large.add_argument('--zoom', type=int, default=19, help='Zoom level (19 for tight crops)')
    p_large.add_argument('--padding', type=float, default=0.5, help='Padding around caravan (0.5 = 50%% on each side)')
    p_large.add_argument('--no-tight-crop', action='store_true', help='Disable tight cropping (use fixed size instead)')
    p_large.add_argument('--size', type=int, default=256, help='Fixed image size (only used with --no-tight-crop)')
    p_large.add_argument('--no-filter', action='store_true', help='Disable empty plot filtering (include all images)')

    # Verify mask alignment
    p_verify = subparsers.add_parser('verify', help='Verify mask alignment on training data')
    p_verify.add_argument('--input', required=True, help='Training data directory')
    p_verify.add_argument('--samples', type=int, default=10, help='Number of samples to check')
    p_verify.add_argument('--output', help='Optional output image path (displays if not set)')

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
        result = step3_download_imagery(args.input, args.output, args.source, args.zoom, args.size)
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

    elif args.command == 'large':
        result = step_large_dataset(
            args.output,
            target_count=args.count,
            source=args.source,
            zoom=args.zoom,
            tight_crop=not args.no_tight_crop,
            padding=args.padding,
            size=args.size,
            filter_empty=not args.no_filter
        )
        return 0 if result else 1

    elif args.command == 'verify':
        verify_mask_alignment(args.input, args.samples, args.output)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
