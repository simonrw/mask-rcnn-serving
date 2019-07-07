#!/usr/bin/env python

from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import tensorflow as tf
from tensorflow.python.framework import graph_util
import shutil


class MyConfig(Config):
    NAME = "serving-testing"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81


def save_model(model, output_dir):
    session = tf.keras.backend.get_session()

    # Have to initialise all of the variables before saving
    session.run(tf.global_variables_initializer())

    shutil.rmtree(output_dir, ignore_errors=True)
    inputs = {t.name: t for t in model.keras_model.input}
    outputs = {t.name: t for t in model.keras_model.output}

    tf.compat.v1.saved_model.simple_save(
        session, output_dir, inputs=inputs, outputs=outputs
    )


if __name__ == "__main__":
    import argparse
    import os

    default_output_dir = os.path.join(os.path.dirname(__file__), "models", "maskrcnn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-dir", required=False, default=default_output_dir
    )
    parser.add_argument("--model-version", required=False, default="1")
    args = parser.parse_args()

    model_dir = os.path.join(args.output_dir, args.model_version)

    model = MaskRCNN(mode="inference", config=MyConfig(), model_dir=model_dir)
    model.load_weights("Mask_RCNN/mask_rcnn_coco.h5", by_name=True)

    save_model(model, model_dir)
