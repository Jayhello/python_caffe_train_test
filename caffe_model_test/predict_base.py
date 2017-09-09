# _*_coding:utf-8 _*_

import sys
sys.path.insert(0, '../../caffe_train_test/')
import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from utils.img_process import *


class CaffePredict(object):

    def __init__(self, b_gpu, mean_path, deploy_path, model_path):

        if b_gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        mean_blob = caffe_pb2.BlobProto()
        with open(mean_path) as f:
            mean_blob.ParseFromString(f.read())

        mean_array = np.asarray(mean_blob.data, dtype=np.float32).\
            reshape((mean_blob.channels, mean_blob.height, mean_blob.width))

        self.net = caffe.Net(deploy_path, model_path, caffe.TEST)

        # Define image transformers
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', mean_array)
        self.transformer.set_transpose('data', (2, 0, 1))

    def predict(self, img):
        img = pre_process_img(img)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        out = self.net.forward()
        pred_probas = out['prob']

        # predict result
        ret_lst = [round(f, 4) for f in pred_probas[0].tolist()]
        return ret_lst


def get_default_caffe_predict():
    # Read model architecture and trained model's weights
    mean_path = "/data6/light/storm_1_1/images/ad_train_py/input_data/mean.binaryproto"
    deploy_path = "/data6/light/storm_1_1/images/ad_train_py/caffe_model/caffenet_deploy_1.prototxt"
    model_path = "/data6/light/storm_1_1/images/ad_train_py/caffe_model/caffe_model_1_iter_10000.caffemodel"
    b_gpu = True
    caffe_predict = CaffePredict(b_gpu, mean_path, deploy_path, model_path)
    return caffe_predict


if __name__ == '__main__':
    caffe_predict = get_default_caffe_predict()

    img_path = '/data6/light/storm_1_1/images/ad_train_py/test_data/0.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print caffe_predict.predict(img)

    pass
