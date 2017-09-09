import sys
sys.path.insert(0, '../../caffe_train_test/')
from predict_base import CaffePredict, get_default_caffe_predict
import glob
import cv2


def get_img_lst(img_dir):
    """
    img_dir: /data6/light/storm_1_1/images/ad_train_py/test_data/
    lots of images like '0.jpg, 1.jpg ......'
    """
    return glob.glob(img_dir + "*.png")


def predict_all():
    path = '/data6/light/storm_1_1/images/ad_train_py/test_data/'
    img_lst = get_img_lst(path)
    caffe_predict = get_default_caffe_predict()

    for path in img_lst:
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # caffe_predict.predict is not thread safe,so can't be used in multiple thread
            # python is dummy multiple threads
            ret_lst = caffe_predict.predict(img)
            print path, ret_lst
        except Exception, e:
            print e


if __name__ == '__main__':
    predict_all()
    pass
