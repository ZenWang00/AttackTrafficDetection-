#-*- encoding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
#from inpudata import Dataset
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
#
TP = pd.read_csv("./TestPrediction/TP.csv")
#
def datafeature(f_name):
	file_value = f_name.values
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature

def discard_fiv_tupple(data):
    for i in range(10):
        # protoc
        data[:, 7 + i * 160] = 0
        # ip and port
        data[:, 10 + i * 160:22 + i * 160] = 0
    return data

#parameter
classes_num = 2
img_shape = 40*40
_batch_size=2000
batch_size = tf.placeholder(tf.int32,[])
#cnn net
_X = tf.placeholder(tf.float32,[None,img_shape],name='x_input')

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

#40*40*1
cnn_input = tf.reshape(_X,[-1,40,40,1])
_X = tf.placeholder(tf.float32,[None,img_shape],name='x_input')
y = tf.placeholder(tf.int32,[None,classes_num])
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conv_1 =  tf.nn.relu(conv2d(cnn_input,W_conv1) + b_conv1)
#36*36*32
pool_1 = max_pool(conv_1)
#18*18*32

W_conv2 = weight_variable([3,3,32,64])
b_conv2 = bias_variable([64])
conv_2 = tf.nn.relu(conv2d(pool_1,W_conv2) + b_conv2)
#16*16*64
pool_2 = max_pool(conv_2)
#8*8*64 = 4096

W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
pool_2_flat = tf.reshape(pool_2,[-1,8*8*64])
cnn_fc1 = tf.matmul(pool_2_flat,W_fc1) + b_fc1
cnn_fc1_drop = tf.nn.dropout(cnn_fc1,keep_prob)

W_fc2 = weight_variable([1024,classes_num])
b_fc2 = bias_variable([classes_num])
logits = tf.matmul(cnn_fc1_drop,W_fc2) + b_fc2
y = tf.placeholder(tf.int32,[None,classes_num])
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

y = tf.one_hot(indices=tf.argmax(input=y,axis=1),depth=classes_num,dtype="int32")

predictions = {
      "classes": tf.argmax(input=logits, axis=1,name='classes'),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

mydata = datafeature(TP)
mydata = discard_fiv_tupple(mydata)
x_raw = np.array(mydata[:,:],dtype="float32")

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('tmp/model2.ckpt.meta')
  new_saver.restore(sess, 'tmp/model2.ckpt')
  sess.run(tf.initialize_all_variables())
  graph = tf.get_default_graph()

  _X = graph.get_tensor_by_name('x_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  predictions["classes"] = graph.get_tensor_by_name('classes:0')

  result = sess.run(predictions["classes"], feed_dict={_X:x_raw,keep_prob:1.0})
  print(result)
  pd.DataFrame(result).to_csv('./PredictionResult.csv')

