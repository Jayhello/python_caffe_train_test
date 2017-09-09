# _*_coding:utf-8 _*_

import sys
sys.path.insert(0, '../../caffe_train_test/')
import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

from utils.img_process import *


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMG_HEIGHT,
        height=IMG_WIDTH,
        label=label,
        data=np.rollaxis(img, 2).tostring())


class GenerateLmdb(object):

    def __init__(self, img_path):
        """
        img_path -> multiple calss directory
        like, class_1, class_2, class_3....
        each class has corresponding class image like class_1_1.png
        :param img_path:
        """
        # get all the images in different class directory
        self.img_lst = glob.glob(os.path.join(img_path, '*', '*.png'))
        print 'input_img list num is %s' % len(self.img_lst)
        # shuffle all the images
        random.shuffle(self.img_lst)

    def generate_lmdb(self, label_lst, percentage, train_path, validation_path):
        """
        label_lst like ['class_1', 'class_2', 'class_3', .....]
        percentage like is 5 (4/5) then 80% be train image, (1/5) 20% be validation image
        train_path like that '/data/train/train_lmdb'
        validation_path like '/data/train/validation_lmdb'
        """
        print 'now generate train lmdb'
        self._generate_lmdb(label_lst, percentage, True, train_path)
        print 'now generate validation lmdb'
        self._generate_lmdb(label_lst, percentage, False, validation_path)

        print '\n generate all images'

    def _generate_lmdb(self, label_lst, percentage, b_train, input_path):
        """
        b_train is True means to generate train lmdb, or validation lmdb
        """
        output_db = lmdb.open(input_path, map_size=int(1e12))
        with output_db.begin(write=True) as in_txn:
            for idx, img_path in enumerate(self.img_lst):

                # create train data
                if b_train:
                    # !=0 means validation data then skip loop
                    if idx % percentage != 0:
                        continue
                # create validation data
                else:
                    # ==0 means train data then skip
                    if idx % percentage == 0:
                        continue

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = pre_process_img(img)
                # path like that '../../class_1/0001.png'
                # so img_path.split('/')[-2] -> class_1
                label = label_lst.index(img_path.split('/')[-2])
                datum = make_datum(img, label)
                in_txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
                print '{:0>5d}'.format(idx) + '->label: ', label, " " + img_path

        output_db.close()


def get_label_lst_by_dir(f_dir):
    """
    f_dir like 'home/user/class', sub dir 'class_1', 'class_2'...'class_n'
    :return: ['class_1', 'class_2'...'class_n']
    """
    return os.listdir(f_dir)

if __name__ == '__main__':
    img_path = '../../ad_train/'
    cl = GenerateLmdb(img_path)

    train_lmdb = '/data6/light/storm_1_1/images/ad_train_py/input_data/train_lmdb'
    validation_lmdb = '/data6/light/storm_1_1/images/ad_train_py/input_data/validation_lmdb'

    os.system('rm -rf  ' + train_lmdb)
    os.system('rm -rf  ' + validation_lmdb)

    input_path = '/data6/light/storm_1_1/images/ad_train/'
    label_lst = get_label_lst_by_dir(input_path)
    print 'label_lst is: %s' % ', '.join(label_lst)

    # (1/10)10% to be validation data, 90% to be train data
    percentage = 10

    cl.generate_lmdb(label_lst, percentage, train_lmdb, validation_lmdb)

    pass
