"""
Caravan Binary Classifier - Colab Training Code

Based on ONS methodology for caravan detection.
Copy each section into separate Colab cells.
"""

# =============================================================================
# CELL 1: Setup and Imports
# =============================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import zipfile
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""

# =============================================================================
# CELL 2: Upload and Extract Patches
# =============================================================================
"""
print("Upload your patches.zip file")
print("Structure should be: patches/caravan/*.png and patches/not_caravan/*.png")
uploaded = files.upload()

zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('.')

# Find the patches directory
import glob
caravan_patches = glob.glob('**/caravan/*.png', recursive=True)
not_caravan_patches = glob.glob('**/not_caravan/*.png', recursive=True)

print(f"Found {len(caravan_patches)} caravan patches")
print(f"Found {len(not_caravan_patches)} non-caravan patches")

# Determine data directory
DATA_DIR = os.path.dirname(os.path.dirname(caravan_patches[0]))
print(f"Data directory: {DATA_DIR}")
"""

# =============================================================================
# CELL 3: Dataset Class
# =============================================================================
"""
class CaravanPatchDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, train_split=0.85):
        self.transform = transform

        # Load all patch paths
        caravan_dir = os.path.join(data_dir, 'caravan')
        not_caravan_dir = os.path.join(data_dir, 'not_caravan')

        caravan_files = [os.path.join(caravan_dir, f) for f in os.listdir(caravan_dir) if f.endswith('.png')]
        not_caravan_files = [os.path.join(not_caravan_dir, f) for f in os.listdir(not_caravan_dir) if f.endswith('.png')]

        # Create labels (1 = caravan, 0 = not caravan)
        all_files = [(f, 1) for f in caravan_files] + [(f, 0) for f in not_caravan_files]

        # Shuffle with fixed seed for reproducibility
        import random
        random.seed(42)
        random.shuffle(all_files)

        # Split into train/val
        split_idx = int(len(all_files) * train_split)
        if train:
            self.data = all_files[:split_idx]
        else:
            self.data = all_files[split_idx:]

        print(f"{'Train' if train else 'Val'} set: {len(self.data)} patches")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
"""

# =============================================================================
# CELL 4: LeNet-style CNN Model
# =============================================================================
"""
class CaravanClassifier(nn.Module):
    '''
    LeNet-5 inspired architecture for caravan classification.
    Input: 48x48 RGB patches (or adjust PATCH_SIZE)
    Output: Binary classification (caravan / not caravan)
    '''
    def __init__(self, patch_size=48):
        super(CaravanClassifier, self).__init__()

        self.features = nn.Sequential(
            # Conv1: 3 -> 32 channels, 5x5 kernel
            nn.Conv2d(3, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2

            # Conv2: 32 -> 64 channels, 5x5 kernel
            nn.Conv2d(32, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2

            # Conv3: 64 -> 128 channels, 3x3 kernel
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size
        # For 48x48: after conv1+pool = 22x22, after conv2+pool = 9x9, after conv3 = 7x7
        # 7 * 7 * 128 = 6272
        self._calc_flat_size(patch_size)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _calc_flat_size(self, patch_size):
        # Simulate forward pass to get size
        x = torch.zeros(1, 3, patch_size, patch_size)
        x = self.features(x)
        self.flat_size = x.view(1, -1).size(1)
        print(f"Flattened feature size: {self.flat_size}")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(1)
"""

# =============================================================================
# CELL 5: Training Setup
# =============================================================================
"""
PATCH_SIZE = 48  # Must match your extracted patches
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),  # Caravans can be any orientation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CaravanPatchDataset(DATA_DIR, transform=train_transform, train=True)
val_dataset = CaravanPatchDataset(DATA_DIR, transform=val_transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Create model
model = CaravanClassifier(patch_size=PATCH_SIZE).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
"""

# =============================================================================
# CELL 6: Training Loop
# =============================================================================
"""
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100 * correct / total, all_preds, all_labels


# Training loop
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_caravan_classifier.pth')

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
"""

# =============================================================================
# CELL 7: Plot Training Curves
# =============================================================================
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train')
ax1.plot(val_losses, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

ax2.plot(train_accs, label='Train')
ax2.plot(val_accs, label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
"""

# =============================================================================
# CELL 8: Confusion Matrix
# =============================================================================
"""
# Load best model
model.load_state_dict(torch.load('best_caravan_classifier.pth'))
_, _, preds, labels = validate(model, val_loader, criterion, device)

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Caravan', 'Caravan'],
            yticklabels=['Not Caravan', 'Caravan'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(labels, preds, target_names=['Not Caravan', 'Caravan']))
"""

# =============================================================================
# CELL 9: Download Model
# =============================================================================
"""
# Save model with metadata
save_dict = {
    'model_state_dict': model.state_dict(),
    'patch_size': PATCH_SIZE,
    'best_val_acc': best_val_acc,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}
torch.save(save_dict, 'caravan_classifier_complete.pth')

files.download('caravan_classifier_complete.pth')
print("Model downloaded!")
"""
