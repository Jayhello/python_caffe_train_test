# _*_coding:utf-8 _*_

import cv2
import urllib
import numpy as np

IMG_HEIGHT = 227
IMG_WIDTH = 227


def pre_process_img(img, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    # firstly histogram equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # resize image to size
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def get_cv_img__from_url(url):
    """
    read image from url to cv codec
    :param url:
    :return:
    """
    try:
        url_response = urllib.urlopen(url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    except Exception, e:
        print e
        return None


if __name__ == '__main__':
    url = 'http://www.sanyarb.com.cn/images/attachement/jpg/site2/20161009/A121475977636942_change_ljx6a9_b.jpg'
    img = get_cv_img__from_url(url)
    cv2.imshow("zhan lang", img)

    img = pre_process_img(img)
    cv2.imshow("pre_process_img", img)
    cv2.waitKey()
    pass
