import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score



def data2feature(f_name, cla):
    file_value = f_name.values
    file_value[:, -1] = cla
    feature = file_value
    feature = feature[:, 1:]
    np.random.shuffle(feature)
    return feature


def discard_fiv_tupple(data):
    for i in range(10):
        data[7 + i * 160] = 0
        data[10 + i * 160:22 + i * 160] = 0
    return data

Benign = pd.read_csv("../flow_labeled/labeld_Monday-Benign.csv")  # 339621

DoS_GoldenEye = pd.read_csv("../flow_labeled/labeld_DoS-GlodenEye.csv")  # 7458

# Heartbleed = pd.read_csv("../flow_labeled/labeld_Heartbleed-Port.csv")#1

DoS_Hulk = pd.read_csv("../flow_labeled/labeld_DoS-Hulk.csv")  # 14108

DoS_Slowhttps = pd.read_csv("../flow_labeled/labeld_DoS-Slowhttptest.csv")  # 4216

DoS_Slowloris = pd.read_csv("../flow_labeled/labeld_DoS-Slowloris.csv")  # 3869

SSH_Patator = pd.read_csv("../flow_labeled/labeld_SSH-Patator.csv")  # 2511

FTP_Patator = pd.read_csv("../flow_labeled/labeld_FTP-Patator.csv")  # 3907

Web_Attack_BruteForce = pd.read_csv("../flow_labeled/labeld_WebAttack-BruteForce.csv")  # 1353
Web_Attack_SqlInjection = pd.read_csv("../flow_labeled/labeld_WebAttack-SqlInjection.csv")  # 12
Web_Attack_XSS = pd.read_csv("../flow_labeled/labeld_WebAttack-XSS.csv")  # 631

# Infiltraton = pd.read_csv()#3

Botnet = pd.read_csv("../flow_labeled/labeld_Botnet.csv")  # 1441

PortScan_1 = pd.read_csv("../flow_labeled/labeld_PortScan_1.csv")  # 344
PortScan_2 = pd.read_csv("../flow_labeled/labeld_PortScan_2.csv")  # 158329  > 158673

DDoS = pd.read_csv("../flow_labeled/labeld_DDoS.csv")  # 16050

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


Data = np.concatenate(Data_tupple, axis=0)
Data = discard_fiv_tupple(Data)
np.random.shuffle(Data)

for x in range(10):
    Data[:, 10 + 160 * x:21 + 160 * x] = 0

x_raw = np.array(Data[:, :-1], dtype="float32")
y_raw = np.array(Data[:, -1], dtype="int32")

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.25, random_state=0)

#X_train,X_test,y_train,y_test=train_test_split(data.iloc[:,0:1600],data.iloc[:,1600],test_size=0.2)

train_data=lgb.Dataset(data_train,label=label_train)
validation_data=lgb.Dataset(data_test,label=label_test)
params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':4,
    'min_data_in_leaf':20,
    'objective':'multiclass',
    'num_class':11,  #lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
}
test_start = time.time()
clf=lgb.train(params,train_data,valid_sets=[validation_data])

y_pred=clf.predict(data_test)
y_pred=[list(x).index(max(x)) for x in y_pred]
#print(y_pred)
#print(accuracy_score(label_test,y_pred))

print("\nModel report")
print(auc)
print("\ntest cost time :%d" %(time.time() - test_start))
print("\nAccuracy:%f" %metrics.accuracy_score(label_test,y_pred))
#print("\nPrecision:%f" %metrics.average_precision_score(label_test,y_pred))
#print("\nRecall:%f" %metrics.recall_score(label_test,y_pred))
#print("\nF1-score:%f" %metrics.f1_score(label_test,y_pred))
#print("\nconfusion matrix:" )
#print("\n%s" %metrics.confusion_matrix(label_test,y_pred))