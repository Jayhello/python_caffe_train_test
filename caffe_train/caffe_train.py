# _*_ coding:utf-8

import os


def get_train_cmd(caffe_path, solver_path, log_path):
    # create train command
    return '%s train --solver %s |& tee %s ' % (caffe_path, solver_path, log_path)


if __name__ == '__main__':

    caffe_path = "/home/xiongyu/caffe/build/tools/caffe"
    solver_path = "/home/xiongyu/caffe_models/caffe_model_1/solver_1.prototxt"
    log_path = "/home/xiongyu/caffe_models/caffe_model_1/model_1_train.log"

    train = get_train_cmd(caffe_path, solver_path, log_path)

    print train
    # use caffe to train model
    os.system(train)

    pass
