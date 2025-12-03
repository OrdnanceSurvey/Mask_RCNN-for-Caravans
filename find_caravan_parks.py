#!/usr/bin/env python3
"""
Find Caravan Parks with Good OSM Coverage

Searches OpenStreetMap for caravan parks in the UK that have many
individually mapped caravans - these are the best candidates for
creating a fully-labeled training dataset.

Usage:
    python find_caravan_parks.py --top 50
    python find_caravan_parks.py --region wales --top 20
"""

import argparse
import requests
import json
from collections import defaultdict
import time

# UK regions with bounding boxes
UK_REGIONS = {
    "uk": "-10.5,49.5,2.0,61.0",  # All of UK
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


def query_caravan_parks(bbox, verbose=True):
    """
    Query OSM for caravan parks and their caravans.

    Returns list of parks with caravan counts.
    """
    west, south, east, north = map(float, bbox.split(","))

    if verbose:
        print(f"Querying OSM for region: {west},{south} to {east},{north}")

    # Query for caravan sites AND individual static caravans
    overpass_query = f"""
    [out:json][timeout:180];
    (
        // Caravan sites (the parks themselves)
        way["tourism"="caravan_site"]({south},{west},{north},{east});
        relation["tourism"="caravan_site"]({south},{west},{north},{east});

        // Individual static caravans
        way["building"="static_caravan"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    overpass_url = "https://overpass-api.de/api/interpreter"

    try:
        if verbose:
            print("Sending query to Overpass API (this may take a minute)...")
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=180)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        return []

    if verbose:
        print(f"Received {len(data.get('elements', []))} elements")

    # Parse results
    nodes = {}
    parks = []
    caravans = []

    for element in data.get('elements', []):
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])
        elif element['type'] == 'way':
            tags = element.get('tags', {})

            # Get coordinates
            coords = []
            for node_id in element.get('nodes', []):
                if node_id in nodes:
                    coords.append(nodes[node_id])

            if not coords:
                continue

            # Calculate centroid
            center_lon = sum(c[0] for c in coords) / len(coords)
            center_lat = sum(c[1] for c in coords) / len(coords)

            # Calculate bounding box
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]

            item = {
                'id': element['id'],
                'name': tags.get('name', 'Unnamed'),
                'center': (center_lon, center_lat),
                'bbox': (min(lons), min(lats), max(lons), max(lats)),
                'tags': tags
            }

            if tags.get('tourism') == 'caravan_site':
                parks.append(item)
            elif tags.get('building') == 'static_caravan':
                caravans.append(item)

    if verbose:
        print(f"Found {len(parks)} caravan parks and {len(caravans)} individual caravans")

    return parks, caravans


def find_caravans_in_parks(parks, caravans, buffer=0.002, verbose=True):
    """
    Match caravans to parks based on location.

    buffer: degrees of lat/lon buffer around park bbox (~200m)
    """
    if verbose:
        print("Matching caravans to parks...")

    park_caravans = defaultdict(list)
    unmatched_caravans = []

    for caravan in caravans:
        c_lon, c_lat = caravan['center']
        matched = False

        for park in parks:
            p_west, p_south, p_east, p_north = park['bbox']

            # Check if caravan is within park bbox (with buffer)
            if (p_west - buffer <= c_lon <= p_east + buffer and
                p_south - buffer <= c_lat <= p_north + buffer):
                park_caravans[park['id']].append(caravan)
                matched = True
                break

        if not matched:
            unmatched_caravans.append(caravan)

    # Add caravan counts to parks
    for park in parks:
        park['caravan_count'] = len(park_caravans.get(park['id'], []))
        park['caravans'] = park_caravans.get(park['id'], [])

    if verbose:
        print(f"Matched {sum(len(v) for v in park_caravans.values())} caravans to parks")
        print(f"Unmatched caravans (potential clusters): {len(unmatched_caravans)}")

    return parks, unmatched_caravans


def find_caravan_clusters(caravans, cluster_radius=0.005, min_cluster_size=5, verbose=True):
    """
    Find clusters of caravans that aren't in mapped parks.

    These might be parks that aren't mapped as tourism=caravan_site
    but have individual caravans mapped.

    cluster_radius: ~500m in degrees
    """
    if verbose:
        print("Finding caravan clusters...")

    if not caravans:
        return []

    # Simple clustering: group caravans within radius
    clusters = []
    used = set()

    for i, caravan in enumerate(caravans):
        if i in used:
            continue

        cluster = [caravan]
        used.add(i)
        c_lon, c_lat = caravan['center']

        for j, other in enumerate(caravans):
            if j in used:
                continue
            o_lon, o_lat = other['center']

            # Simple distance check
            if abs(c_lon - o_lon) < cluster_radius and abs(c_lat - o_lat) < cluster_radius:
                cluster.append(other)
                used.add(j)

        if len(cluster) >= min_cluster_size:
            # Calculate cluster center and bbox
            lons = [c['center'][0] for c in cluster]
            lats = [c['center'][1] for c in cluster]

            clusters.append({
                'id': f"cluster_{i}",
                'name': f"Unmapped cluster ({len(cluster)} caravans)",
                'center': (sum(lons)/len(lons), sum(lats)/len(lats)),
                'bbox': (min(lons), min(lats), max(lons), max(lats)),
                'caravan_count': len(cluster),
                'caravans': cluster,
                'is_cluster': True
            })

    if verbose:
        print(f"Found {len(clusters)} caravan clusters")

    return clusters


def format_results(parks, clusters, top_n=50):
    """Format and sort results."""
    # Combine parks and clusters
    all_locations = parks + clusters

    # Sort by caravan count
    all_locations.sort(key=lambda x: x['caravan_count'], reverse=True)

    # Take top N
    return all_locations[:top_n]


def print_results(locations, output_file=None):
    """Print results in a nice format."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TOP CARAVAN PARKS/CLUSTERS BY NUMBER OF MAPPED CARAVANS")
    lines.append("=" * 80 + "\n")

    for i, loc in enumerate(locations, 1):
        is_cluster = loc.get('is_cluster', False)
        type_str = "[CLUSTER]" if is_cluster else "[PARK]"

        lines.append(f"{i:3}. {type_str} {loc['name']}")
        lines.append(f"     Caravans mapped: {loc['caravan_count']}")
        lines.append(f"     Center: {loc['center'][1]:.5f}, {loc['center'][0]:.5f}")
        lines.append(f"     OSM: https://www.openstreetmap.org/?mlat={loc['center'][1]:.5f}&mlon={loc['center'][0]:.5f}&zoom=17")
        if not is_cluster:
            lines.append(f"     Way: https://www.openstreetmap.org/way/{loc['id']}")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nResults saved to: {output_file}")

    return locations


def save_json(locations, output_file):
    """Save results as JSON for further processing."""
    # Clean up for JSON serialization
    export = []
    for loc in locations:
        export.append({
            'id': loc['id'],
            'name': loc['name'],
            'caravan_count': loc['caravan_count'],
            'center_lat': loc['center'][1],
            'center_lon': loc['center'][0],
            'bbox': loc['bbox'],
            'osm_link': f"https://www.openstreetmap.org/?mlat={loc['center'][1]:.5f}&mlon={loc['center'][0]:.5f}&zoom=17",
            'is_cluster': loc.get('is_cluster', False)
        })

    with open(output_file, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"JSON saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find caravan parks with good OSM coverage for labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find top 50 parks in all of UK
    python find_caravan_parks.py --top 50

    # Find parks in Wales only
    python find_caravan_parks.py --region wales --top 20

    # Save results to files
    python find_caravan_parks.py --top 30 --output parks.txt --json parks.json

Available regions: """ + ", ".join(UK_REGIONS.keys())
    )

    parser.add_argument('--region', default='uk', choices=UK_REGIONS.keys(),
                        help='Region to search (default: uk)')
    parser.add_argument('--bbox', help='Custom bbox: "west,south,east,north"')
    parser.add_argument('--top', type=int, default=50, help='Number of top results (default: 50)')
    parser.add_argument('--output', help='Save text results to file')
    parser.add_argument('--json', help='Save JSON results to file')
    parser.add_argument('--min-caravans', type=int, default=5,
                        help='Minimum caravans to include (default: 5)')

    args = parser.parse_args()

    # Get bounding box
    bbox = args.bbox if args.bbox else UK_REGIONS[args.region]

    print(f"\nSearching for caravan parks in: {args.region}")
    print(f"Bounding box: {bbox}\n")

    # Query OSM
    parks, caravans = query_caravan_parks(bbox)

    if not parks and not caravans:
        print("No results found. Try a different region or check your connection.")
        return 1

    # Match caravans to parks
    parks, unmatched = find_caravans_in_parks(parks, caravans)

    # Find clusters of unmatched caravans
    clusters = find_caravan_clusters(unmatched, min_cluster_size=args.min_caravans)

    # Filter parks with minimum caravans
    parks = [p for p in parks if p['caravan_count'] >= args.min_caravans]

    # Format and print results
    top_locations = format_results(parks, clusters, args.top)
    print_results(top_locations, args.output)

    # Save JSON if requested
    if args.json:
        save_json(top_locations, args.json)

    print(f"\nTotal locations found: {len(top_locations)}")
    print(f"Total caravans in top locations: {sum(l['caravan_count'] for l in top_locations)}")

    return 0


if __name__ == "__main__":
    exit(main())
