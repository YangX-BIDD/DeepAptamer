#%%
import tensorflow as tf
import keras
import re
import keras_tuner
from keras.regularizers import *
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import ipykernel
#import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as  aps
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D,Conv2D, Dense, MaxPooling1D,MaxPooling2D, Flatten, LSTM, Input, Attention, MultiHeadAttention
    
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
#from keras.layers.merge import Concatenate
TF_ENABLE_ONEDNN_OPTS=0
#%%
#weightdir = '/home/yangx/DeepSELEX/script/Classification/'
def create_deepselex():
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=8
                    , strides=1
                    , kernel_initializer='RandomNormal'
                    , activation = 'relu'
                    , kernel_regularizer = l2(5e-3)
                    , use_bias=True
                    , bias_initializer='RandomNormal'
                    , input_shape=(35, 4)))
    #model.add(MaxPooling1D(pool_size=4))
    #model.add(Conv1D(filters=32, kernel_size=12, 
    #             input_shape=(35, 4)))
    model.add(MaxPooling1D(pool_size=5, strides=None
                           , padding = 'valid'
                           , data_format = 'channels_last'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))


    #model.summary()
    Adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999
                                , decay=1e-5,
                                amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer='adam'
                    #, metrics=['accuracy']
                    )
    return model
def model_deepselex_run(model_path, X_train, y_train, serialize_dir,model,batch_size=64):
    param = model.summary()
    print(param)
    history = model.fit(X_train, y_train
                    , batch_size = batch_size
                    , epochs=30
                    , verbose=1
                    , shuffle=True
                    , validation_split=0.3
                    , callbacks=[keras.callbacks.ModelCheckpoint(model_path
                        , monitor='val_loss'
                        , verbose=0, save_best_only=True
                        , save_weights_only=False, mode='auto'
                        , period=1),
                        keras.callbacks.EarlyStopping(monitor='val_loss'
                        , min_delta=0
                        , patience=1
                        , verbose=0, mode='auto', restore_best_weights=True)
                                ])

    model_json = model.to_json()
    with open(serialize_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(serialize_dir+"model.h5")
    print("Saved model to disk")
    return history

# %%
def evaluate(model,X_test,y_test,resultdir,dnn_type):
    y_score = model.predict(X_test)
    plt.rcParams['font.size'] = 14
    Font={'size':43, 'family':'Arial'}
    cm = confusion_matrix(np.argmax(y_test, axis=1), 
                      np.argmax(y_score, axis=1))
    print('Confusion matrix:\n',cm)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(dnn_type,Font)
    plt.colorbar()

    plt.ylabel('True label',fontdict=Font)
    plt.xlabel('Predicted label',Font)
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 fontdict=Font,
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
    
    plt.savefig(resultdir+'confusion_matrix.png')
    plt.show()
    plt.clf()
    
    Font={'size':18, 'family':'Arial'}
    fpr, tpr, thersholds = roc_curve(y_test.T[1], y_score.T[1], pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(6,6))
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr,label = dnn_type+' = %0.3f' % roc_auc, color='RoyalBlue')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    plt.title('ROC curve',Font)
    #plt.legend(loc='best')
    #plt.show()   
    #plt.savefig(resultdir+'ROC.png')
    #plt.clf() 
    plt.legend(loc = 'lower right', prop=Font)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=18)
    plt.savefig(resultdir+'ROC.png')
    plt.show()
    return y_test, y_score,fpr,tpr,roc_auc
#%%
class deepselex:
    def __init__(self, test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath):
        self.test_ratio,self.inputfile,self.seqfile,self.pos_num,self.neg_num,self.outputpath = test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath
    
    def data_process(self):
        onehot_fea = []
        shape_fea = []
        label_pos=float(self.pos_num)
        label_neg=float(self.neg_num)
        #seq = pd.read_csv(datadir+"sequence.txt",header=None)
        with open(self.inputfile) as sequences:
            for line in sequences:
                line = re.split("\t|\n",line)
                onehot = [float(x) for x in line[2:142]]
                onehot_file = []
                for i in range(0,140,4):
                    onehot_file.append(onehot[i:i+4])
                onehot_fea.append(onehot_file)

                shap = line[142:-1]
                #shap.append(line[0])
                shape = [float(x) for x in shap]
                shape_fea.append(shape)
            
        onehot_features = np.array(onehot_fea[:int(label_pos)]+onehot_fea[-1*int(label_neg):])
        shape_features = np.array(shape_fea[:int(label_pos)]+shape_fea[-1*int(label_neg):])

        label_pos = int(label_pos)#*len(se))
        label_neg = int(label_neg)#*len(se))
        labels = []
        labels = ['1']*int(label_pos)+ ['0']*int(label_neg)
        one_hot_encoder = OneHotEncoder(categories='auto')
        labels = np.array(labels).reshape(-1, 1)
        input_labels = one_hot_encoder.fit_transform(labels).toarray()
        print('Labels:\n',labels.T)
        print('One-hot encoded labels:\n',input_labels.T)
        self.onehot_feature,self.input_label,self.shape_feature=onehot_features,input_labels,shape_features
    
    #%%
    def data_sample(self,pos,neg):
        self.pos,self.neg=pos,neg
        self.onehot_features=np.concatenate((self.onehot_feature[:pos],self.onehot_feature[neg:]))
        self.input_labels=np.concatenate((self.input_label[:pos],self.input_label[neg:]))
        self.shape_features=np.concatenate((self.shape_feature[:pos],self.shape_feature[neg:]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
    def model_deepselex(self):
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        #self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        #BATCH_SIZE = 32
        self.resultdir = self.outputpath
        #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
        
        self.model_selex = create_deepselex()
        #model_cnn.load_weights(self.resultdir+'cnn/model.h5')
        self.history_selex = model_deepselex_run(self.resultdir+'model.h5', self.X_train, self.y_train, self.resultdir,self.model_selex,640)
        #create_plots(self.model_cnn,self.history_cnn,self.resultdir+'cnn/')
        self.y_label_selex, self.y_score_selex, self.fpr_selex, self.tpr_selex, self.roc_auc_selex = evaluate(self.model_selex,self.X_test,self.y_test,self.resultdir+'deepselex/','DeepSELEX')
    
    
    def all_metrics(self,target):

        self.roc_auc1,self.roc_auc2,self.roc_auc3=ks(target,self.resultdir,self.y_score_cnn,self.y_label_cnn,"CNN",self.y_score_lstm,self.y_label_lstm,"BiLSTM",self.y_score_cnnlstm,self.y_label_cnnlstm,"DeepAptamer")  
        self.pr_auc,self.precision,self.recall=ks_pr(target,self.resultdir,self.y_score_cnn,self.y_label_cnn,"CNN",self.y_score_lstm,self.y_label_lstm,"BiLSTM",self.y_score_cnnlstm,self.y_label_cnnlstm,"DeepAptamer")  
        self.f1_score1=f1_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round(),pos_label=1)
        self.f1_score2=f1_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round(),pos_label=1)
        self.f1_score3=f1_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round(),pos_label=1)
        self.mmc1=mcc(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())
        self.mmc2=mcc(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
        self.mmc3=mcc(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
        self.precision=[metrics.precision_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())\
                        ,metrics.precision_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
                        ,metrics.precision_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
                        ]
        self.recall=[metrics.recall_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())
                     ,metrics.recall_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
                     ,metrics.recall_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
                    ]
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import rcParams
        import matplotlib as mpl
        sns.set_theme(style="white",font='Arial',font_scale=1.4)
        #custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        #sns.set_theme(style="ticks", rc=custom_params)
        mpl.rcParams["font.family"] = 'Arial'
        mpl.rcParams["mathtext.fontset"] = 'cm' 
        mpl.rcParams["axes.linewidth"] = 2
        font = {'family':'Arial','size':45}
        mpl.rc('font',**font)
        #mpl.rc('legend',**{'fontsize':45})
        mpl.rcParams['savefig.bbox'] = 'tight'
        font = {'family' : 'Arial','weight' : 'bold'}  
        plt.rc('font', **font) 
        Font={'size':18, 'family':'Arial'}
        self.metrics_df=pd.DataFrame({"Performance":[self.roc_auc1,self.roc_auc2,self.roc_auc3]
                                 +self.pr_auc
                                 +self.precision
                                 +self.recall
                                 +[self.f1_score1,self.f1_score2,self.f1_score3]
                                 +[self.mmc1,self.mmc2,self.mmc3],
                                   "Metrics":["AUROC"]*3
                                            +["AUPRC"]*3
                                            +["Precision"]*3
                                            +["Recall"]*3
                                            +["F1_Score"]*3
                                            +["MCC"]*3
                                ,"Models":["CNN","BiLSTM","DeepAptamer"]*6
                                            })
        
        plt.figure(figsize=(11,8))
        fig = sns.barplot(x ="Metrics", y = 'Performance', data = self.metrics_df, hue = "Models",palette=sns.color_palette("Paired"))
        fig.legend(loc='center right', bbox_to_anchor=(.8,1.05), ncol=3,fontsize=16)
        #fig.legend(ncol=4)
        fig.set_xticklabels( ["AUROC" ,
                                             "AUPRC" ,
                                             "Precision" ,
                                             "Recall" ,
                                             "F1_Score" ,
                                             "MCC" ], fontsize=14,weight='bold')
        fig.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontsize=14,weight='bold')
        plt.ylabel('Performance', Font,weight='bold')
        plt.xlabel(target, Font,weight='bold')
        plt.grid(False)
        
        plt.savefig(self.resultdir+'all_metircs.png')
# %%
def evaluate_svc(model,X_test,y_test,resultdir,dnn_type):
    y_score = model.predict_proba(X_test)
    plt.rcParams['font.size'] = 14
    Font={'size':43, 'family':'Arial'}
    cm = confusion_matrix(y_test, 
                      np.argmax(y_score,axis=1))
    print('Confusion matrix:\n',cm)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(dnn_type,Font)
    plt.colorbar()

    plt.ylabel('True label',fontdict=Font)
    plt.xlabel('Predicted label',Font)
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 fontdict=Font,
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
    
    plt.savefig(resultdir+'confusion_matrix.png')
    plt.show()
    plt.clf()
    
    Font={'size':18, 'family':'Arial'}
    fpr, tpr, thersholds = roc_curve(y_test, y_score.T[1], pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(6,6))
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr,label = dnn_type+' = %0.3f' % roc_auc, color='RoyalBlue')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    plt.title('ROC curve',Font)
    #plt.legend(loc='best')
    #plt.show()   
    #plt.savefig(resultdir+'ROC.png')
    #plt.clf() 
    plt.legend(loc = 'lower right', prop=Font)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=18)
    plt.savefig(resultdir+'ROC.png')
    plt.show()
    return y_test, y_score,fpr,tpr,roc_auc


class svc:
    def __init__(self, test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath):
        self.test_ratio,self.inputfile,self.seqfile,self.pos_num,self.neg_num,self.outputpath = test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath
    

    def data_process(self):
    
        Exp = pd.read_csv(self.inputfile,header=None,sep='\t')
        data = Exp[Exp.columns[2:]]
        data = pd.concat([data[:int(self.pos_num)],data[-int(self.neg_num):]])
        data['affinity'] = [1]*int(self.pos_num)+[0]*int(self.neg_num)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(data.iloc[:,:-1]), np.array(data['affinity']), test_size=self.test_ratio, random_state=42)
    
    def model_svc(self):
        self.resultdir=self.outputpath
        self.clf = SVC(kernel = "rbf"
              ,gamma="auto"
              ,degree = 1
              ,cache_size = 5000
              ,probability=True
             ).fit(self.X_train, self.y_train)
        self.y_label_svc, self.y_score_svc, self.fpr_svc, self.tpr_svc, self.roc_auc_svc = evaluate_svc(self.clf,self.X_test,self.y_test,self.resultdir+'svc/','SVM')
        #score = clf.predict(X_test)
        #svc_fpr, svc_tpr, thresholds = roc_curve(float(y_test),float(score),pos_label=1)
        #self.y_score_svc = clf.predict(self.X_test)
# %%
