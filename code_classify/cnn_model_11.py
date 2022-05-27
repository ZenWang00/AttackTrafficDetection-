#!/usr/bin/python
#-*- encoding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
#from inpudata import Dataset
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import confusion_matrix

#============================================================================
class Dataset:
    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]
        pass
    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._data = self.data[idx0]
            self._label = self.label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            label_new_part =  self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._label[start:end]

def data_prepare(f1_name,f2_name,y1,y2):
	d1 = f1_name.values
	d2 = f2_name.values
	d1[:,-1] = y1
	d2[:,-1] = y2

	dataset = np.concatenate((d1,d2),axis=0)

	
	np.random.shuffle(dataset)
	return dataset

def data2feature(f_name,cla):
	file_value = f_name.values
	file_value[:,-1] = cla
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature
def discard_fiv_tupple(data):
	
	for i in range(10):
		#protoc
		data[:,7+i*160] = 0
		#ip and port
		data[:,10+i*160:22+i*160] = 0
	return data

start = time.time()

Benign = pd.read_csv("../flow_labeled/labeld_Monday-Benign.csv")#339621

DoS_GoldenEye = pd.read_csv("../flow_labeled/labeld_DoS-GlodenEye.csv")#7458

# Heartbleed = pd.read_csv("../flow_labeled/labeld_Heartbleed-Port.csv")#1

DoS_Hulk = pd.read_csv("../flow_labeled/labeld_DoS-Hulk.csv")#14108

DoS_Slowhttps = pd.read_csv("../flow_labeled/labeld_DoS-Slowhttptest.csv")#4216

DoS_Slowloris = pd.read_csv("../flow_labeled/labeld_DoS-Slowloris.csv")#3869

SSH_Patator = pd.read_csv("../flow_labeled/labeld_SSH-Patator.csv")#2511

FTP_Patator = pd.read_csv("../flow_labeled/labeld_FTP-Patator.csv")#3907

Web_Attack_BruteForce = pd.read_csv("../flow_labeled/labeld_WebAttack-BruteForce.csv")#1353
Web_Attack_SqlInjection = pd.read_csv("../flow_labeled/labeld_WebAttack-SqlInjection.csv")#12
Web_Attack_XSS = pd.read_csv("../flow_labeled/labeld_WebAttack-XSS.csv")#631

# Infiltraton = pd.read_csv()#3

Botnet = pd.read_csv("../flow_labeled/labeld_Botnet.csv")#1441

PortScan_1 = pd.read_csv("../flow_labeled/labeld_PortScan_1.csv")#344
PortScan_2 = pd.read_csv("../flow_labeled/labeld_PortScan_2.csv")#158329  > 158673

DDoS = pd.read_csv("../flow_labeled/labeld_DDoS.csv")#16050


print("\ndataset prepared,cost time:%d" %(time.time() - start))

d0 = data2feature(Benign,0)[:18000]
d1 = data2feature(DoS_GoldenEye,1)
d2 = data2feature(DoS_Hulk,2)
d3 = data2feature(DoS_Slowhttps,3)
d4 = data2feature(DoS_Slowloris,4)
d5 = data2feature(SSH_Patator,5)
d6 = data2feature(FTP_Patator,6)

d7 = data2feature(Web_Attack_BruteForce,7)
d8 = data2feature(Web_Attack_SqlInjection,7)
d9 = data2feature(Web_Attack_XSS,7)

d10 = data2feature(Botnet,8)

d11 = data2feature(PortScan_1,9)
d12 = data2feature(PortScan_2,9)[:15000]

d13 = data2feature(DDoS,10)

Data_tupple = (d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13)

Data = np.concatenate(Data_tupple,axis=0)

Data = discard_fiv_tupple(Data)

np.random.shuffle(Data)

x_raw = np.array(Data[:,:-1],dtype="float32")
x_raw = discard_fiv_tupple(x_raw)
y_raw = np.array(Data[:,-1],dtype="int32")
#
#data_total = pd.DataFrame(x_raw)
#data_total.to_csv('data_total.csv')
#data_flag = pd.DataFrame(y_raw)
#data_flag.to_csv('data_flag.csv')

data_train,data_test,label_train,label_test = train_test_split(x_raw,y_raw,test_size=0.2,random_state=0)

#==========================================================================



#==========================================================================
def labels_transform(mlist,classes):
	
	batch_label = np.zeros((len(mlist),classes),dtype="i4")
	for i in range(len(mlist)):
		batch_label[i][mlist[i]] = 1
	return batch_label
#============================================================================

#parameter
learning_rate = 0.0005
img_shape = 40*40
classes_num = 11
batch_size = tf.placeholder(tf.int32,[])
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 20000

# cnn network

_X = tf.placeholder(tf.float32,[None,img_shape])
y = tf.placeholder(tf.int32,[None,classes_num])
keep_prob = tf.placeholder(tf.float32)

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
#

W_fc2 = weight_variable([1024,classes_num])
b_fc2 = bias_variable([classes_num])
logits = tf.matmul(cnn_fc1_drop,W_fc2) + b_fc2

predictions = {
	"classes":tf.argmax(input=logits,axis=1,name=""),
	"probabilities":tf.nn.softmax(logits,name="softmax_tensor")
	}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
y = tf.one_hot(indices=tf.argmax(input=y,axis=1),depth=classes_num,dtype="int32")
loss = tf.losses.softmax_cross_entropy(y,logits)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,).minimize(loss)

correct_prediction = tf.equal(predictions["classes"],tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

TP = tf.metrics.true_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

sess = tf.Session()
saver = tf.train.Saver()

print("\n"+"="*50 +"Benign Trainging"+"="*50)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
_batch_size = 128
#
mydata_train = Dataset(data_train,label_train)
statr = time.time()
for i in range(train_iter):
	batch = mydata_train.next_batch(_batch_size)
	labels = labels_transform(batch[1],classes_num)
	if (i+1)%200 ==0:

		train_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:labels,
			keep_prob:1.0,batch_size:_batch_size})
		
		print("\nthe %dth loop,training accuracy:%f" %(i+1,train_accuracy))

	sess.run(train_op,feed_dict={_X:batch[0],y:labels,keep_prob:0.5,
		batch_size:_batch_size})

#save_path = saver.save(sess, "./tmp1/model11.ckpt")
#print("\ntraining finished cost time:%f" %(time.time() - statr))



test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
test_batch_size = 2000
preLabel = []
mlabel = []
test_iter = len(data_test)//test_batch_size + 1

mydata_test = Dataset(data_test,label_test)

print("\n"+"="*50+"Benign test"+"="*50)
test_start = time.time()
for i in range(test_iter):
	batch = mydata_test.next_batch(test_batch_size)
	#df = pd.DataFrame(batch)
	#df.to_csv('./model11test/mydata_test.csv')
	mlabel = mlabel + list(batch[1])
	labels = labels_transform(batch[1],classes_num)
	df = pd.DataFrame(batch[0])
	df.to_csv('./model11test/demo'+str(i)+'.csv')
	e_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_tp,value_tp = sess.run(TP,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_fp,value_fp = sess.run(FP,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_tn,value_tn = sess.run(TN,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_fn,value_fn = sess.run(FN,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	preLabel = preLabel + list(sess.run(predictions["classes"],feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size}))


	test_accuracy = test_accuracy + e_accuracy
	true_positives = true_positives + value_tp
	false_positives = false_positives + value_fp
	true_negatives = true_negatives + value_tn
	false_negatives = false_negatives + value_fn
	
print("\ntest cost time :%d" %(time.time() - test_start))
print("\n"+"="*50+"Test result"+"="*50)
print("\n test accuracy :%f" %(test_accuracy/test_iter))
print("\n true positives :%d" %true_positives)
print("\n false positives :%d" %false_positives)
print("\n true negatives :%d" %true_negatives)
print("\n false negatives :%d" %false_negatives)
print("\n"+"="*50+"  DataSet Describe  "+"="*50)
print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" %(len(x_raw),len(data_train),len(data_test)))


mP = true_positives/(true_positives+false_positives)
mR = true_positives/(true_positives+false_negatives)
mF1_score = 2*mP*mR/(mP+mR)

print("\nPrecision:%f" %mP)
print("\nRecall:%f" %mR)
print("\nF1-Score:%f" %mF1_score)
conmat = confusion_matrix(mlabel,preLabel)
print("\nConfusion Matraics:")
#print(conmat)
#print(len(mlabel))

