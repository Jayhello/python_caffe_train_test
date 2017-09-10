# project framework
1. caffe_model:the training and deploy prototxt files

2. train_data_generate:generate training lmdb, validation lmdb, and mean_binaryproto

3. caffe_train: training caffe model

4. caffe_model_test:test model recognition results, both local files, and files from database

5. utils:image process fucntion, url(of image) to cv2 format, database process

### The directory -> caffe model
------------

[caffe Alexnet training file train_val.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt), change the input lmdb path

[parameters files solver.prototxt ](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/solver.prototxt), change input path

[deploy file deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt) change output_num like training prototxt files


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/caffe_traintxt_1.jpg)


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/caffe_traintxt_2.jpg)


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/caffe_traintxt_3.jpg)


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/caffe_traintxt_4.jpg)


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/caffe_traintxt_5.jpg)


### utils
------------

read image from url coded in cv2 format


![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/img_url_2%20(2).jpg)


### caffe_train.py, create_mean_binaryproto.py
------------
generate train command and run train

create_mean_binaryproto.py create mean binary proto file

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/mean_1.png)

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/mean_2.png)

### caffe model test
------------

base predict class

The below demo predict one image

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/predict_base_1.jpg)

The below demo predict images from local directory

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/predict_from_local_1.jpg)


The below demo predict images from database and write recognition results to database

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/predict_from_db_1.jpg)

![](https://github.com/Jayhello/python_caffe_train_test/blob/master/run_images/predict_from_db_2.jpg)


-----------------

You can see all the explantion in this [bolg](http://blog.csdn.net/haluoluo211/article/details/77918156)


Cited some contents from the below two articles.

>http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

>https://software.intel.com/en-us/articles/training-and-deploying-deep-learning-networks-with-caffe-optimized-for-intel-architecture

