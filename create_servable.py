#!/usr/bin/env python

from mrcnn.model import MaskRCNN
from mrcnn.config import Config


class MyConfig(Config):
    NAME = "serving-testing"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81

model = MaskRCNN(mode="inference", config=MyConfig(), model_dir="models")
model.load_weights("Mask_RCNN/mask_rcnn_coco.h5", by_name=True)
