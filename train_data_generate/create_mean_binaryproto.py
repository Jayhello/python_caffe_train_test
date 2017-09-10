# _*_ coding:utf-8

import os


def get_mean_cmd(mean_tool_path, train_lmdb_path, mean_binaryproto_path):
    # create train command
    return '%s -backend=lmdb %s %s ' % (mean_tool_path, train_lmdb_path, mean_binaryproto_path)


if __name__ == '__main__':
    mean_tool_path = '/home/ubuntu/caffe/build/tools/compute_image_mean'
    train_lmdb_path = '/home/xiongyu/input/train_lmdb'
    mean_binaryproto_path = '/home/xiongyu/input/mean.binaryproto'

    cmd = get_mean_cmd(mean_tool_path, train_lmdb_path, mean_binaryproto_path)
    print cmd

    os.system(cmd)
