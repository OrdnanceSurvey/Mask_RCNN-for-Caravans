# Detectron2 Caravan Detection Setup

This guide will help you set up Detectron2 for caravan detection on Windows with Python 3.10.

## Step 1: Create a New Virtual Environment

```bash
# Create new environment (separate from the old one)
python -m venv detectron2_env

# Activate it
# Windows:
detectron2_env\Scripts\activate
# Linux/Mac:
source detectron2_env/bin/activate
```

## Step 2: Install PyTorch

First, check your CUDA version:
```bash
nvidia-smi
```

Then install PyTorch with matching CUDA version:

```bash
# For CUDA 11.8 (most common)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower, but works)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Step 3: Install Detectron2

### Windows Installation

On Windows, build from source:

```bash
# Install build dependencies
pip install ninja yacs cython

# Install pycocotools
pip install pycocotools

# Install detectron2 from GitHub
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you get build errors on Windows, try:
```bash
# Alternative: install pre-built wheel (if available for your Python/CUDA version)
# Check: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

### Linux Installation (easier)

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Step 4: Install Other Dependencies

```bash
pip install opencv-python numpy pillow matplotlib
```

## Step 5: Verify Installation

```python
python -c "import detectron2; print(detectron2.__version__)"
```

## Usage

### Training

```bash
# Navigate to the detectron2_caravans folder
cd detectron2_caravans

# Train from COCO pre-trained weights
python train_caravans.py train --dataset "C:\Users\Fatima.Khan\data\large_dataset\training"

# With custom output directory
python train_caravans.py train --dataset "C:\Users\Fatima.Khan\data\large_dataset\training" --output ./my_caravan_model

# Resume training from checkpoint
python train_caravans.py train --dataset "C:\Users\Fatima.Khan\data\large_dataset\training" --resume
```

### Detection/Inference

```bash
# Detect on single image
python train_caravans.py detect --weights ./output_caravan/model_final.pth --image /path/to/image.tif

# Detect on folder of images
python train_caravans.py detect --weights ./output_caravan/model_final.pth --folder /path/to/images/

# With custom threshold and save masks
python train_caravans.py detect --weights ./output_caravan/model_final.pth --folder /path/to/images/ --threshold 0.7 --save-masks
```

## Dataset Format

Your dataset should be structured like this:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image001.tif
│   │   ├── image002.tif
│   │   └── ...
│   └── labels/
│       └── 1/
│           ├── image001.png
│           ├── image002.png
│           └── ...
└── val/
    ├── images/
    │   └── ...
    └── labels/
        └── 1/
            └── ...
```

**Mask format:** Each pixel value > 0 represents a unique caravan instance. Value 0 is background.

## Configuration Options

You can modify `train_caravans.py` to adjust:

- `SOLVER.IMS_PER_BATCH`: Batch size (reduce if out of GPU memory)
- `SOLVER.BASE_LR`: Learning rate
- `SOLVER.MAX_ITER`: Number of training iterations
- `MODEL.ROI_HEADS.SCORE_THRESH_TEST`: Detection confidence threshold
- `INPUT.MIN_SIZE_TRAIN`: Input image sizes for training

## Troubleshooting

### Out of Memory

Reduce batch size in `train_caravans.py`:
```python
cfg.SOLVER.IMS_PER_BATCH = 1  # Instead of 2
```

### Slow Training

- Make sure CUDA is being used: `torch.cuda.is_available()` should return `True`
- Increase number of workers: `cfg.DATALOADER.NUM_WORKERS = 8`

### Windows Build Errors

If Detectron2 won't build on Windows:
1. Install Visual Studio Build Tools 2019+
2. Make sure you have the "Desktop development with C++" workload
3. Try using WSL2 (Windows Subsystem for Linux) instead

## Output

After training, you'll find:
- `output_caravan/model_final.pth` - Final trained model
- `output_caravan/model_*.pth` - Periodic checkpoints
- `output_caravan/metrics.json` - Training metrics
- `output_caravan/events.*` - TensorBoard logs

View training progress with TensorBoard:
```bash
tensorboard --logdir output_caravan
```
