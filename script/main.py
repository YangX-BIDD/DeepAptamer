#%%
import tensorflow as tf
import keras
import re
#import keras_tuner
from keras.regularizers import *
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import ipykernel
#import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as  aps
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D,Conv2D, Dense, MaxPooling1D,MaxPooling2D, Flatten, LSTM, Input

import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import itertools
#import pydot
import getopt
import sys
import numpy as np
from sklearn import metrics
import pylab as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Bidirectional,Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from focal_loss import BinaryFocalLoss
import sys
sys.path.insert(0,'~/DeepSELEX/DeepApta/script/')  
from model import *
TF_ENABLE_ONEDNN_OPTS=0

# %%
datadir = "/home/yangx/DeepSELEX/train_data/"
weightdir = '/home/yangx/DeepSELEX/DeepApta/model/'
def tune(model_name,pos,neg_init,target):
    for i in range(neg_init,150000,10000):
        neg=-1*i
        model_name.data_sample(pos,neg)
   
        model_name.model(256,100)
        #model.model_load()
        model_name.all_metrics(target)
        if model_name.roc_auc3>model_name.roc_auc2+0.02>model_name.roc_auc1+0.02 and model_name.pr_auc3>model_name.pr_auc2+0.02>model_name.pr_auc1+0.02:
            print(model_name.metrics_df)
            return model_name.lables
            break
        else:
            continue

#%%
CT_deepapta=deepapta(0.1,datadir+'CT_20.txt',datadir+'CT-20.txt','150000','300000',weightdir+'CTGF/9k_1w/')
CT_deepapta.data_process()
#%%
sample_count_CT=tune(CT_deepapta,7500,30000,'CTGF')
#neg=30000
#%%

pos,neg=9000,-10000
#deepapta.data_sample=data_sample
CT_deepapta.data_sample(pos,neg)
#%%

#deepapta.model=model
CT_deepapta.model(256,100,100)
CT_deepapta.all_metrics("CTGF")
#CT_deepapta.model_load()
#%%
deepapta.model_load=model_load
deepapta.all_metrics=all_metrics
CT_deepapta.model_load(100,'9k_1w/')
CT_deepapta.all_metrics('CTGF')
#%%
def tune_layer(model_name,layer_init,target):
    for i in range(layer_init,200,10):
        
        model_name.model_combine(256,100,i)
   
        model_name.all_metrics(target)
        if model_name.roc_auc3>model_name.roc_auc2+0.02>model_name.roc_auc1+0.02 and model_name.pr_auc3>model_name.pr_auc2+0.02>model_name.pr_auc1+0.02:
            print(model_name.metrics_df)
            return model_name.lables
            break
        else:
            continue
layer_CT=tune_layer(CT_deepapta,120,'CTGF')
#%%




#%%

DK_deepapta=deepapta(0.1,datadir+'DK_30.txt',datadir+'DK-30.txt','150000','300000',weightdir+'DKK1/')
DK_deepapta.data_process()
#%%
#smaple_count_DK = tune(DK_deepapta,4000,40000,'DKK1')

#%%

pos,neg=4000,-10000

DK_deepapta.data_sample(pos,neg)
#DK_deepapta.model_cnnbilstm_attention(128,100,50)

DK_deepapta.model(256,100,100)
#%%

DK_deepapta.model_cnnbilstm_attention(128,100,100)
#%%
DK_deepapta.all_metrics('DKK1')
#%%
#layer_DK=tune_layer(DK_deepapta,100,'DKK1')
#%%
deepapta.model_combine=model_combine
DK_deepapta.model_combine(128,100,500)
DK_deepapta.all_metrics('DKK1')
#DK_deepapta.model_load('best_model/')

#%%

DK_deepapta.all_metrics('DKK1')
#%%
deepapta.model=model
DK_deepapta.model(256,100,200)
DK_deepapta.all_metrics('DKK1')



#%%
BC_deepapta=deepapta(0.1,datadir+'BC_6.txt',datadir+'BC-6.txt','150000','300000',weightdir+'BCMA/')
BC_deepapta.data_process()
#%%
#smaple_count_BC=tune(BC_deepapta,2000,40000,'BCMA')
#%%
pos,neg=4000,-10000
BC_deepapta.data_sample(pos,neg)
#deepapta.model=model
BC_deepapta.model(256,100,100)
#%%
BC_deepapta.all_metrics('BCMA')

# %%
BC_deepapta.model(256,100,100)
layer_BC=tune_layer(BC_deepapta,100,'BCMA')
# %%
from script.comparison_models import *
# %%
DK_deepselex=deepselex(0.1,datadir+'DK_30.txt',datadir+'DK-30.txt','150000','300000',weightdir+'DKK1/')
DK_deepselex.data_process()
# %%
pos,neg=4000,-7000

DK_deepselex.data_sample(pos,neg)
#%%
#deepselex.model_selex=model_selex
DK_deepselex.model_deepselex()
#%%
from comparison_models import *
DK_svc=svc(0.1,datadir+'DK_30.txt',datadir+'DK-30.txt','4000','7000',weightdir+'DKK1/')
DK_svc.data_process()

DK_svc.model_svc()
#%%
DK_svc.y_score_svc
# %%
deepselex.all_metrics=all_metrics
DK_deepselex.all_metrics('DKK1')
# %%
DK_deepselex.y_score_cnn
# %%
DK_svc.clf.predict_proba(DK_svc.X_train)
# %%
DK_svc.y_train
# %%
