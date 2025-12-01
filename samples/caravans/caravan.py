"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

# Try to import augmentation library (optional)
try:
    import imgaug.augmenters as iaa
    HAS_IMGAUG = True
except ImportError:
    HAS_IMGAUG = False
    print("Note: imgaug not available. Training will proceed without augmentation.")

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class TopoConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "caravan"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    GPU_COUNT = 1

    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # Background + Buildings + Roads + Water

    # Number of training steps per epoch
    # CHANGE BACK TO 100!!
    STEPS_PER_EPOCH = 315
    VALIDATION_STEPS = 20
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # LEARNING_RATE = 0.0001

    # DELETE THIS WHEN ACTUALLY RUNNING!
    # ORIGINALLY 50
    # VALIDATION_STEPS = 20


############################################################
#  Dataset
############################################################

class TopoDataset(utils.Dataset):

    def load_topo(self, dataset_dir, subset):
        """Load a subset of topolite dataset from the given dataset_dir"""
        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        #add classes
        self.add_class("caravan", 1, "1")
        

        
        # loading images
        self._image_dir = os.path.join(dataset_dir, "images//")
        self._mask_dir = os.path.join(dataset_dir, "labels//")
        i = 0
        for f in glob.glob(os.path.join(self._image_dir, "*.tif")):
            filename = os.path.split(f)[1]
            # print(filename)
            self.add_image("caravan", image_id=i, path=f,
                           width=Config.IMAGE_MAX_DIM, height=Config.IMAGE_MAX_DIM, filename=filename)
            i += 1
            
    def load_mask(self, image_id):
        """Read instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        maskdir = self._mask_dir
        # print("\n \n \n")
        # print(maskdir)
        # print("\n \n \n")
        fname = info["filename"]
        masks = []
        class_ids = []
         #looping through all the classes, loading and processing corresponding masks
        for ci in self.class_info:
            class_name = ci["name"]
            class_id = ci["id"]
            # print("WE ARE HERE")
            try:
                print(self._mask_dir)
                class_path = os.path.join(self._mask_dir, "1")
                fname = fname.replace('.tif', '.png')
                m_src = skimage.io.imread(os.path.join(self._mask_dir, class_name, fname))
            except:
                #no file with masks of this class found
                continue                
            #making individual masks for each instance
                   

            instance_ids = np.unique(m_src)
            for i in instance_ids:
                if i > 0:
                    m = np.zeros(m_src.shape)
                    m[m_src==i] = i
                    if np.any(m==i):
                        masks.append(m)
                        class_ids.append(class_id)
        try:
            masks = np.stack(masks, axis=-1)        
        except:
            print("!!! no masks were found.", info)
            
        # Return mask, and array of class IDs of each instance.
        return masks.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "caravan":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def image_mask(image, mask):
    # import cv2
    # h,w,c = image.shape
    print("performing image_mask")
    black = np.zeros_like(image)
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        result = np.where(mask,image,black).astype(np.uint8)
    else:
        result = black.astype(np.uint8)
    return result

def create_class_mask(model, image_path=None, video_path=None, class_name="road"):
    assert image_path
    assert class_name
    print("performing create_class_mask")
    MASK_DIR = "masks"

    imagename = os.path.basename(os.path.normpath(image_path))
    imageid = imagename.split(".")[0]

    print("imageid: {}".format(imageid))

    classdict = {"BuiltEnv": 1, "MadeSurface": 2, "NaturalLow": 3, "Rail": 4, "Vegetation": 5, "Water": 6}
    classid = classdict[class_name]

    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    black = np.zeros_like(image)
    white = 255 - black
    # Detect Objects
    r = model.detect([image], verbose=1)[0]
    class_ids = r['class_ids']
    idxs =  [i for i,x in enumerate(class_ids) if x == classid]
    masks = np.dstack((r['masks'][:,:,i] for i in idxs))
    mask = image_mask(white, r['masks'])
    # Save output
    file_name = "{}_{}.tif".format(class_name, imageid)
    skimage.io.imsave(os.path.join(MASK_DIR, file_name), mask)
    print("Saved to: {}".format(file_name))


def batch_class_mask(model, folder_path=None, video_path=None, class_name="road"):
    assert folder_path
    assert class_name
    print("performing batch_mask")
    
    MASK_DIR = "masks"
    classdict = {"BuiltEnv": 1, "MadeSurface": 2, "NaturalLow": 3, "Rail": 4, "Vegetation": 5, "Water": 6}
    classid = classdict[class_name]
    ROOT_DIR = os.path.dirname(folder_path)
    print(ROOT_DIR)  
    OUTPUT_DIR = os.path.join(ROOT_DIR, "masks")

    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tif"):
            imagename = file
            print(file)
            imageid = imagename.split(".")[0]
            im_path = os.path.join(folder_path, file)
            print("Creating mask for: {}".format(im_path))
            print("imageid: {}".format(imageid))

            # Read image
            image = skimage.io.imread(im_path)
            black = np.zeros_like(image)
            white = 255 - black
            # Detect Objects
            r = model.detect([image], verbose=1)[0]
            class_ids = r['class_ids']
            idxs =  [i for i,x in enumerate(class_ids) if x == classid]
            if len(idxs) > 0:
                masks = np.dstack((r['masks'][:,:,i] for i in idxs))
                mask = image_mask(white, masks)
                # the above should be image_mask(white, masks)
                # Save output
                file_name = "{}_{}.tif".format(class_name, imageid)
                skimage.io.imsave(os.path.join(OUTPUT_DIR, file_name), mask)
                print("Saved to: {}".format(OUTPUT_DIR + "\\" + file_name))
            else:
                print("No instances found!")
    print("\n COMPLETE \n")



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TopoDataset()
    dataset_train.load_topo(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TopoDataset()
    dataset_val.load_topo(args.dataset, "val")
    dataset_val.prepare()

    # Data augmentation pipeline for aerial imagery
    # This helps when training with small datasets (e.g., 50-100 images)
    augmentation = None
    if HAS_IMGAUG:
        augmentation = iaa.Sequential([
            # Horizontal and vertical flips (caravans can face any direction)
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # Random 90-degree rotations (aerial view has no fixed orientation)
            iaa.Sometimes(0.5, iaa.Rot90([1, 2, 3])),
            # Small rotations for more variety
            iaa.Sometimes(0.3, iaa.Affine(rotate=(-15, 15))),
            # Brightness and contrast adjustments (different lighting conditions)
            iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.3, iaa.LinearContrast((0.8, 1.2))),
        ])
        print("Using imgaug for data augmentation")
    else:
        print("Training without augmentation (imgaug not installed)")

    # Training - Stage 1: Train heads only
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2: Fine-tune all layers
    print("Fine-tuning all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=50,
                layers='all',
                augmentation=augmentation)    



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    # CHANGE THIS BACK TO TRUE!!
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--classtype', required=False,
                        metavar="class type (e.g. building, road, water)",
                        help='Class type to create mask of')
    parser.add_argument('--folder', required=False,
                        metavar="image folder",
                        help="folder where images comes from")

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image,\
               "Provide --image or --video to apply color splash"
    elif args.command == "class_mask":
        assert args.image or args.video,\
               "Provide --image to apply class mask"
        assert args.classtype,\
                "Provide --class to apply class mask"
    elif args.command == "batch_class_mask":
        assert args.folder,\
                "Provide --folder to apply class mask"
        assert args.classtype,\
                "Provide --class to apply class mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TopoConfig()
    else:
        class InferenceConfig(TopoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        # UNCOMMENT THIS!!
        model.load_weights(weights_path, by_name=True)
        # pass

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    elif args.command == "mask":    
        print("Now we output the mask")
        create_mask(model, image_path=args.image)
    elif args.command == "class_mask":
        print("Now we output the {} mask".format(args.classtype))
        create_class_mask(model, image_path=args.image, class_name=args.classtype)
    elif args.command == "batch_class_mask":
        print("Now we output the {} mask folder".format(args.classtype))
        batch_class_mask(model, folder_path=args.folder, class_name=args.classtype)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash' or 'mask'".format(args.command))
