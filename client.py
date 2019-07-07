#!/usr/bin/env python


import sys

sys.path.append("protos")
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

channel = grpc.insecure_channel("127.0.0.1:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

req = predict_pb2.PredictRequest()
req.model_spec.name = "maskrcnn"
req.model_spec.signature_name = "serving_default"

result_future = stub.Predict.future(req, 5.0)
result = result_future.result()
