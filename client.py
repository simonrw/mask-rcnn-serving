#!/usr/bin/env python


import sys
import numpy as np
import time
import tensorflow as tf
import imageio

sys.path.append("protos")
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
import grpc


def make_tensor_proto(img):
    tensor_shape = [1] + list(img.shape)
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape]
    tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_FLOAT,
        tensor_shape=tensor_shape,
        float_val=list(img.reshape(-1)),
    )
    return tensor


if __name__ == "__main__":

    # Raise the maximum message size to 100MB
    max_message_size = 100 * 1024 * 1024
    options = [
        ("grpc.max_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
    ]
    channel = grpc.insecure_channel("127.0.0.1:8500", options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    image_data = imageio.imread("data/155806.jpg")
    image_height, image_width, _ = image_data.shape

    req = predict_pb2.PredictRequest()
    req.model_spec.name = "maskrcnn"
    req.model_spec.signature_name = "serving_default"
    req.inputs["input_anchors:0"].CopyFrom(make_tensor_proto(np.zeros((2, 4))))
    req.inputs["input_image:0"].CopyFrom(make_tensor_proto(image_data))
    req.inputs["input_image_meta:0"].CopyFrom(make_tensor_proto(np.zeros((93))))

    start_time = time.time()
    result_future = stub.Predict.future(req, 5.0)
    result = result_future.result()
    end_time = time.time()

    print(f"Time taken: {(end_time - start_time) * 1000:.2f}ms")
