#!/usr/bin/env python3
"""
Test script: Download ONE satellite image of a caravan location.

Run this to verify the pipeline works:
    python test_download_one_caravan.py

This will:
1. Query OSM for one caravan in Anglesey, UK
2. Download satellite imagery for that location
3. Save the image so you can inspect it
"""

import requests
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import json
import math

def main():
    print("=" * 60)
    print("TEST: Download one caravan satellite image")
    print("=" * 60)

    # Step 1: Query OSM for caravans in Anglesey
    print("\n[1/3] Querying OpenStreetMap for caravans...")

    bbox = (-4.7, 53.1, -4.0, 53.4)  # Anglesey, UK
    west, south, east, north = bbox

    overpass_query = f"""
    [out:json][timeout:60];
    way["building"="static_caravan"]({south},{west},{north},{east});
    out body 1;
    >;
    out skel qt;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=60
        )
        response.raise_for_status()
        osm_data = response.json()
    except Exception as e:
        print(f"Error querying OSM: {e}")
        print("\nTrying alternative Overpass server...")
        try:
            response = requests.post(
                "https://lz4.overpass-api.de/api/interpreter",
                data={"data": overpass_query},
                timeout=60
            )
            response.raise_for_status()
            osm_data = response.json()
        except Exception as e2:
            print(f"Error with alternative server: {e2}")
            return 1

    # Parse nodes
    nodes = {}
    caravan = None
    for element in osm_data.get('elements', []):
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])
        elif element['type'] == 'way' and 'nodes' in element:
            caravan = element

    if not caravan:
        print("No caravans found in this area. Try a different bbox.")
        return 1

    # Get caravan center
    coords = [nodes[n] for n in caravan['nodes'] if n in nodes]
    if not coords:
        print("Could not get caravan coordinates")
        return 1

    center_lon = sum(c[0] for c in coords) / len(coords)
    center_lat = sum(c[1] for c in coords) / len(coords)

    print(f"Found caravan at: {center_lat:.6f}, {center_lon:.6f}")
    print(f"OSM ID: {caravan['id']}")

    # Step 2: Download satellite imagery
    print("\n[2/3] Downloading satellite imagery from ESRI...")

    zoom = 19  # High detail
    n = 2 ** zoom
    tile_x = int((center_lon + 180) / 360 * n)
    tile_y = int((1 - np.arcsinh(np.tan(np.radians(center_lat))) / np.pi) / 2 * n)

    # Download 3x3 tiles and stitch
    tiles = []
    for dy in range(-1, 2):
        row = []
        for dx in range(-1, 2):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y + dy}/{tile_x + dx}"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content))
                row.append(tile)
            except Exception as e:
                print(f"Warning: Failed to download tile: {e}")
                row.append(Image.new('RGB', (256, 256), (128, 128, 128)))
        tiles.append(row)

    # Stitch tiles
    stitched = Image.new('RGB', (768, 768))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            stitched.paste(tile, (x * 256, y * 256))

    # Crop to center 512x512
    left = (768 - 512) // 2
    top = (768 - 512) // 2
    final_image = stitched.crop((left, top, left + 512, top + 512))

    # Step 3: Draw polygon and save output
    print("\n[3/3] Drawing polygon and saving output...")

    os.makedirs("data/test_output", exist_ok=True)

    # Calculate the bounding box of the downloaded image in lat/lon
    # Each tile at zoom 19 covers a certain area
    def tile_to_latlon(tile_x, tile_y, zoom):
        """Convert tile coordinates to lat/lon (top-left corner of tile)"""
        n = 2 ** zoom
        lon = tile_x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
        return lat, lon

    # Get bounds of the 3x3 tile grid we downloaded
    top_left_lat, top_left_lon = tile_to_latlon(tile_x - 1, tile_y - 1, zoom)
    bottom_right_lat, bottom_right_lon = tile_to_latlon(tile_x + 2, tile_y + 2, zoom)

    # Adjust for the crop (we cropped from 768x768 to 512x512, removing 128px from each side)
    # 128 pixels = 128/768 of the total span
    crop_ratio = 128 / 768
    lon_span = bottom_right_lon - top_left_lon
    lat_span = top_left_lat - bottom_right_lat  # Note: lat decreases going down

    img_west = top_left_lon + crop_ratio * lon_span
    img_east = bottom_right_lon - crop_ratio * lon_span
    img_north = top_left_lat - crop_ratio * lat_span
    img_south = bottom_right_lat + crop_ratio * lat_span

    print(f"Image bounds: N={img_north:.6f}, S={img_south:.6f}, E={img_east:.6f}, W={img_west:.6f}")

    # Convert caravan polygon coords to pixel coords
    def latlon_to_pixel(lon, lat, img_width=512, img_height=512):
        """Convert lat/lon to pixel coordinates in the image"""
        x = (lon - img_west) / (img_east - img_west) * img_width
        y = (img_north - lat) / (img_north - img_south) * img_height
        return int(x), int(y)

    # Draw the polygon on the image
    draw = ImageDraw.Draw(final_image)

    pixel_coords = [latlon_to_pixel(lon, lat) for lon, lat in coords]
    print(f"Caravan polygon ({len(pixel_coords)} points):")
    for i, (px, py) in enumerate(pixel_coords):
        print(f"  Point {i}: pixel ({px}, {py})")

    # Draw polygon outline in RED
    if len(pixel_coords) >= 3:
        # Draw filled polygon with transparency would require RGBA, so just draw thick outline
        draw.polygon(pixel_coords, outline='red', fill=None)
        # Draw thicker outline by drawing multiple times
        for offset in range(-2, 3):
            shifted = [(x + offset, y) for x, y in pixel_coords]
            draw.polygon(shifted, outline='red', fill=None)
            shifted = [(x, y + offset) for x, y in pixel_coords]
            draw.polygon(shifted, outline='red', fill=None)

    # Save image WITH polygon
    image_path = f"data/test_output/caravan_{caravan['id']}_with_polygon.png"
    final_image.save(image_path)
    print(f"Saved image with polygon: {image_path}")

    # Also save original without polygon
    # Re-download for clean version
    original_image = stitched.crop((left, top, left + 512, top + 512))
    original_path = f"data/test_output/caravan_{caravan['id']}_original.png"
    original_image.save(original_path)
    print(f"Saved original image: {original_path}")

    # Save metadata
    metadata = {
        "osm_id": caravan['id'],
        "center": [center_lon, center_lat],
        "coordinates": coords,
        "pixel_coordinates": pixel_coords,
        "image_bounds": {
            "north": img_north,
            "south": img_south,
            "east": img_east,
            "west": img_west
        },
        "zoom": zoom,
        "source": "ESRI World Imagery"
    }

    metadata_path = "data/test_output/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    print("\n" + "=" * 60)
    print("SUCCESS! Check data/test_output/ folder")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {image_path} (with RED polygon)")
    print(f"  - {original_path} (original)")
    print(f"  - {metadata_path}")
    print(f"\nThe RED polygon shows the caravan boundary from OpenStreetMap.")

    return 0


if __name__ == "__main__":
    exit(main())
