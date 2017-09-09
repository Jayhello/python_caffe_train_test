# _*_ coding:utf-8 _*_

import sys
sys.path.insert(0, '../../caffe_train_test/')
from utils.DbBse import DbService, get_default_db
from utils.img_process import get_cv_img__from_url
from predict_base import CaffePredict, get_default_caffe_predict


def predict_from_db():
    """
    get all the url and id from database and
    then predict, write predict result to database
    :return:
    """
    db = get_default_db()

    # [(1, 'http://xxx.1.jpg'), (2, 'http://xxx.2.jpg).....]
    url_id_lst = db.get_ad_info()

    print 'url_id_lst length is %s: ' % len(url_id_lst)
    print 'url_id_lst first is', url_id_lst[0]

    caffe_predict = get_default_caffe_predict()

    for item in url_id_lst:
        img = get_cv_img__from_url(item[1])
        if img is None:
            continue

        ret_lst = caffe_predict.predict(img)
        # item[0] is id
        ret_lst.append(item[0])
        # write result to database
        print item[1], ret_lst
        db.update_ad_info(ret_lst)


if __name__ == '__main__':
    predict_from_db()
    pass
