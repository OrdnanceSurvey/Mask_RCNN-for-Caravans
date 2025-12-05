"""
Caravan Detection Inference - Sliding Window with Clustering

Based on ONS methodology:
1. Slide classifier across image
2. Build probability heatmap
3. Cluster high-probability regions
4. Output caravan park locations

Copy each section into Colab cells.
"""

# =============================================================================
# CELL 1: Setup (run after training or upload model)
# =============================================================================
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from google.colab import files
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""

# =============================================================================
# CELL 2: Model Definition (same as training)
# =============================================================================
"""
class CaravanClassifier(nn.Module):
    def __init__(self, patch_size=48):
        super(CaravanClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size
        x = torch.zeros(1, 3, patch_size, patch_size)
        x = self.features(x)
        self.flat_size = x.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(1)
"""

# =============================================================================
# CELL 3: Load Model
# =============================================================================
"""
# Upload your trained model if not already present
if not os.path.exists('caravan_classifier_complete.pth'):
    print("Upload your trained model (caravan_classifier_complete.pth)")
    uploaded = files.upload()

# Load model
checkpoint = torch.load('caravan_classifier_complete.pth', map_location=device)
PATCH_SIZE = checkpoint.get('patch_size', 48)
NORMALIZE_MEAN = checkpoint.get('normalize_mean', [0.485, 0.456, 0.406])
NORMALIZE_STD = checkpoint.get('normalize_std', [0.229, 0.224, 0.225])

model = CaravanClassifier(patch_size=PATCH_SIZE).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded! Patch size: {PATCH_SIZE}")
print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}%")

# Transform for inference
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])
"""

# =============================================================================
# CELL 4: Sliding Window Inference
# =============================================================================
"""
def sliding_window_inference(image, model, transform, patch_size=48, stride=16,
                             threshold=0.5, batch_size=256, device='cuda'):
    '''
    Apply classifier using sliding window across image.

    Returns:
        heatmap: Probability map (same size as image)
        detections: List of (x, y, confidence) for high-confidence patches
    '''
    model.eval()

    width, height = image.size
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    # Generate all patches
    patches = []
    positions = []

    half = patch_size // 2
    for y in range(half, height - half, stride):
        for x in range(half, width - half, stride):
            patch = image.crop((x - half, y - half, x + half, y + half))
            patches.append(transform(patch))
            positions.append((x, y))

    print(f"Processing {len(patches)} patches...")

    # Process in batches
    detections = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i:i+batch_size]).to(device)
            probs = model(batch).cpu().numpy()

            for j, prob in enumerate(probs):
                x, y = positions[i + j]

                # Update heatmap (average overlapping predictions)
                y1, y2 = max(0, y - half), min(height, y + half)
                x1, x2 = max(0, x - half), min(width, x + half)
                heatmap[y1:y2, x1:x2] += prob
                counts[y1:y2, x1:x2] += 1

                if prob > threshold:
                    detections.append((x, y, prob))

            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, len(patches))}/{len(patches)} patches")

    # Average overlapping regions
    counts[counts == 0] = 1
    heatmap = heatmap / counts

    print(f"Found {len(detections)} high-confidence patches (threshold={threshold})")

    return heatmap, detections
"""

# =============================================================================
# CELL 5a: Count Individual Caravans (deduplicate overlapping detections)
# =============================================================================
"""
def count_individual_caravans(detections, merge_radius=20):
    '''
    Merge overlapping detections to count individual caravans.

    Since we use a sliding window with stride < patch_size, each caravan
    will trigger multiple overlapping detections. This function merges
    nearby detections into single caravan locations.

    Args:
        detections: List of (x, y, confidence) from sliding window
        merge_radius: Max distance to merge detections (roughly caravan size / 2)

    Returns:
        caravans: List of (x, y, confidence) for each unique caravan
    '''
    if len(detections) == 0:
        return []

    coords = np.array([(d[0], d[1]) for d in detections])
    confidences = np.array([d[2] for d in detections])

    # Use DBSCAN to merge nearby points
    # eps = merge_radius, min_samples = 1 (single detection is enough)
    clustering = DBSCAN(eps=merge_radius, min_samples=1).fit(coords)
    labels = clustering.labels_

    caravans = []
    for label in set(labels):
        if label == -1:
            continue

        mask = labels == label
        cluster_coords = coords[mask]
        cluster_confs = confidences[mask]

        # Use confidence-weighted centroid
        weights = cluster_confs / cluster_confs.sum()
        centroid_x = (cluster_coords[:, 0] * weights).sum()
        centroid_y = (cluster_coords[:, 1] * weights).sum()
        max_conf = cluster_confs.max()

        caravans.append((centroid_x, centroid_y, max_conf))

    print(f"Merged {len(detections)} detections into {len(caravans)} individual caravans")

    return caravans
"""

# =============================================================================
# CELL 5b: Cluster Caravans into Parks
# =============================================================================
"""
def cluster_detections(detections, eps=30, min_samples=3, min_cluster_size=5):
    '''
    Cluster nearby detections using DBSCAN.
    Based on ONS method: only keep clusters with multiple caravans.

    Args:
        detections: List of (x, y, confidence)
        eps: Maximum distance between points in same cluster
        min_samples: Minimum points to form a cluster
        min_cluster_size: Minimum caravans to consider it a "park"

    Returns:
        clusters: List of cluster info dicts
    '''
    if len(detections) < min_samples:
        return []

    # Extract coordinates
    coords = np.array([(d[0], d[1]) for d in detections])
    confidences = np.array([d[2] for d in detections])

    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    # Process each cluster
    clusters = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:  # Noise points
            continue

        mask = labels == label
        cluster_coords = coords[mask]
        cluster_confs = confidences[mask]

        if len(cluster_coords) >= min_cluster_size:
            clusters.append({
                'center': cluster_coords.mean(axis=0),
                'points': cluster_coords.tolist(),
                'num_caravans': len(cluster_coords),
                'avg_confidence': cluster_confs.mean(),
                'bbox': (
                    cluster_coords[:, 0].min(),
                    cluster_coords[:, 1].min(),
                    cluster_coords[:, 0].max(),
                    cluster_coords[:, 1].max()
                )
            })

    # Sort by number of caravans
    clusters.sort(key=lambda x: x['num_caravans'], reverse=True)

    print(f"Found {len(clusters)} caravan clusters (min size={min_cluster_size})")

    return clusters
"""

# =============================================================================
# CELL 6: Visualization
# =============================================================================
"""
def visualize_results(image, heatmap, detections, clusters, figsize=(16, 8)):
    '''Visualize heatmap, detections, and clusters.'''

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image with heatmap overlay
    axes[0].imshow(image)
    heatmap_overlay = axes[0].imshow(heatmap, cmap='jet', alpha=0.4)
    axes[0].set_title('Probability Heatmap')
    axes[0].axis('off')
    plt.colorbar(heatmap_overlay, ax=axes[0], fraction=0.046)

    # Detections
    axes[1].imshow(image)
    if detections:
        xs = [d[0] for d in detections]
        ys = [d[1] for d in detections]
        colors = [d[2] for d in detections]
        scatter = axes[1].scatter(xs, ys, c=colors, cmap='RdYlGn', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=axes[1], fraction=0.046)
    axes[1].set_title(f'Detections ({len(detections)} points)')
    axes[1].axis('off')

    # Clusters with bounding boxes
    axes[2].imshow(image)
    for i, cluster in enumerate(clusters):
        x1, y1, x2, y2 = cluster['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color='red', linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x1, y1-5, f"{cluster['num_caravans']} caravans",
                    color='red', fontsize=8, weight='bold')
    axes[2].set_title(f'Caravan Parks ({len(clusters)} found)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return fig


def draw_detections_on_image(image, clusters, detections=None):
    '''Draw bounding boxes and detection points on image.'''
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Draw individual detections if provided
    if detections:
        for x, y, conf in detections:
            r = 3
            color = (0, int(255 * conf), int(255 * (1-conf)))  # Green = high conf
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    # Draw cluster bounding boxes
    for cluster in clusters:
        x1, y1, x2, y2 = cluster['bbox']
        # Add padding
        pad = 20
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = x2+pad, y2+pad

        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1-15), f"{cluster['num_caravans']} caravans", fill='red')

    return img_draw
"""

# =============================================================================
# CELL 7: Run Inference on Uploaded Image
# =============================================================================
"""
print("Upload an aerial image to analyze:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Load and process image
image = Image.open(image_path).convert('RGB')
print(f"Image size: {image.size}")

# Run sliding window inference
heatmap, detections = sliding_window_inference(
    image, model, transform,
    patch_size=PATCH_SIZE,
    stride=12,           # Smaller = more precise but slower
    threshold=0.6,       # Confidence threshold
    batch_size=512,
    device=device
)

# Count individual caravans (merge overlapping detections)
caravans = count_individual_caravans(
    detections,
    merge_radius=20      # Roughly half a caravan width in pixels
)

# Optionally cluster into parks (for finding caravan park locations)
clusters = cluster_detections(
    caravans,            # Use deduplicated caravans, not raw detections
    eps=60,              # Max distance between caravans in same park
    min_samples=3,       # Min points to form cluster
    min_cluster_size=5   # Min caravans to call it a "park"
)

# Visualize (show individual caravans, not raw overlapping detections)
visualize_results(image, heatmap, caravans, clusters)

# Save annotated image
result_img = draw_detections_on_image(image, clusters, caravans)
result_img.save('result_annotated.png')
print("Saved annotated image to result_annotated.png")

print(f"\\n*** TOTAL CARAVANS DETECTED: {len(caravans)} ***")
"""

# =============================================================================
# CELL 8: Summary Statistics
# =============================================================================
"""
print("\\n" + "="*50)
print("DETECTION SUMMARY")
print("="*50)
print(f"Raw sliding window detections: {len(detections)}")
print(f"Individual caravans (after merging): {len(caravans)}")
print(f"Caravan parks/clusters found: {len(clusters)}")
print()

# Show confidence distribution
if caravans:
    confs = [c[2] for c in caravans]
    print(f"Confidence stats:")
    print(f"  - Min: {min(confs):.2%}")
    print(f"  - Max: {max(confs):.2%}")
    print(f"  - Mean: {np.mean(confs):.2%}")
    print()

for i, cluster in enumerate(clusters):
    print(f"Park/Cluster {i+1}:")
    print(f"  - Caravans: {cluster['num_caravans']}")
    print(f"  - Avg confidence: {cluster['avg_confidence']:.2%}")
    print(f"  - Center: ({cluster['center'][0]:.0f}, {cluster['center'][1]:.0f})")
    print()

print("="*50)
print(f"TOTAL CARAVANS: {len(caravans)}")
print("="*50)
"""

# =============================================================================
# CELL 9: Download Results
# =============================================================================
"""
files.download('result_annotated.png')
"""
