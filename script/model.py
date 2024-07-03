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
def main(argv):
    opts = ''
    inputfile = ''
    seqfile = ''
    pos_num = ''
    neg_num = ''
    outputpath = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h:i:s:p:n:o:', [
                                   'help', 'inputfile=', 'seqfile=','pos_num=','neg_num=', 'outputpath='])
    except getopt.GetoptError:
        print('Command:'+'\n'+'classification.py -i <inputfile> -s <seqfile> -p <pos_num> -n <neg_num> -o <outputpath>'
                  + '\n'+'Result:'+'\n'+'model.h5 & model.json')

        sys.exit()
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print('Command:'+'\n'+'classification.py -i <inputfile> -s <seqfile> -p <pos_num> -n <neg_num> -o <outputpath>'
                  + '\n'+'Result:'+'\n'+'model.h5 & model.json')
            sys.exit(2)
        elif opt in ("-i", "--inputfile"):
            inputfile = arg
        elif opt in ("-s", "--sequence"):
            seqfile = arg  
        elif opt in ("-p", "--pos_num"):
            pos_num = arg 
        elif opt in ("-n", "--neg_num"):
            neg_num = arg 
        elif opt in ("-o", "--outputpath"):
            outputpath = arg

    #print('input: ', inputfile)
    #print('output: ', 'result')
    return inputfile, seqfile, pos_num, neg_num, outputpath

#%%

def data_process(features,label_pos,label_neg):
    onehot_fea = []
    shape_fea = []
    #seq = pd.read_csv(datadir+"sequence.txt",header=None)
    with open(features) as sequences:
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
    return onehot_features,input_labels,shape_features


#%%
def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(35, 4)))
    #model.add(MaxPooling1D(pool_size=4))
    #model.add(Conv1D(filters=32, kernel_size=12, 
    #             input_shape=(35, 4)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    #model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
    model.summary()
    return model



EPCOHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.    
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 1 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.1 # proportion of neurones not used for training
MAXLEN = 150 # cuts text after number of these characters in pad_sequences
checkpoint_dir ='checkpoints'
#%%
def create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    #model.add(Embedding(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, input_length = len(X_train[0]), name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True),input_shape=(35, 4)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='sigmoid'))
    #model.add(Dense(2, activation='relu'))
    
    
    #model.add(Dense(2, activation='sigmoid'))
    
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
#%%
#rnn_hidden_dim = RNN_HIDDEN_DIM
#
#input_dim = INPUT_DIM
def create_combine(gamm,pos_alpha, dropout,layer_num):
    first_inp=Input(shape=(35,4), name='md_1')
    model1 = Sequential()(first_inp)
    model1=Conv1D(filters=12, kernel_size=1, #padding='same',
                 input_shape=(35, 4))(model1)         
    model1=MaxPooling1D(pool_size=1)(model1)
    #model1=Flatten()(model1)
    model1 = Dense(32, activation='relu')(model1)
    model1=Dense(4, activation='softmax')(model1)
    
    
    second_inp=Input(shape=(126,1), name='md_2')
    #second_inp=tf.reshape(second_inp,[21,6,1])
    model2 = Sequential()(second_inp)
    model2=Conv1D(filters=1, kernel_size=100, #padding='same',
                 input_shape=(126,1))(model2)
    model2=MaxPooling1D(pool_size=20)(model2)
    model2 = Dense(4, activation='relu')(model2)
    
    
    #model.compile(loss='binary_crossentropy', optimizer='adam', 
    #             metrics=['binary_accuracy'])    
    #model.add(Embedding(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, input_length = len(X_train[0]), name='embedding_layer'))
    
    #model = Sequential()
    model3=concatenate([model1,model2],axis=1)
    #model3 = Flatten()(model3)
    #model3=Dense(64, activation='relu')(model3)
    
    model3 = Bidirectional(LSTM(units=100, return_sequences=True))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Bidirectional(LSTM(layer_num))(model3)
    model3 = Dropout(dropout)(model3)
    #model3 = Flatten()(model3)
    #model3 = Dense(16, activation='sigmoid')(model3)
    #model3 = Dense(8, activation='relu')(model3)
    #model3 = Dense(8, activation='relu')(model3)
    model3 = Dense(2, activation='sigmoid')(model3)
    #model3 = Dense(2, activation='tanh')(model3)
    ### 最后一层如果是relu，效果会很不好。
    #   tanh还可以，但是不算最好，可以增加epoch后看看
    #   sigmoid也还算可以
    #   droupout从0.8降到0.2后 tpr上升
    #   gamma从3降到2后，tpr再上升
    #   droupout又升到0.6，tpr下降
    #   把relu全部去掉，tpr再次下降
    # 总结，可以考虑先relu，最后tanh+sigmoid，并降低droupout和gamma，并考虑增加正样本量
    
    model=Model(inputs=[first_inp, second_inp], outputs=model3)
    
    '''
    model.compile(optimizer='adam'#tf.keras.optimizers.Adam(learning_rate=0.00001)
                  , loss=#'binary_crossentropy'
                        BinaryFocalLoss(
                                            #from_logits=True
                                            #, apply_class_balancing=True
                                            #,
                                            gamma=gamm
                                            ,pos_weight=pos_alpha
                                            #,alpha=0.9
                                            )
                  #, loss='binary_crossentropy'
                   ,metrics=['accuracy']
                
                )
    
    '''
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
create_combine(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=100)

#%%




#%%
def create_cnnbilstm_attention(gamm,pos_alpha, dropout,layer_num):
    first_inp=Input(shape=(35,4), name='md_1')
    model1 = Sequential()(first_inp)
    model1=Conv1D(filters=12, kernel_size=1, #padding='same',
                 input_shape=(35, 4))(model1)         
    model1=MaxPooling1D(pool_size=1)(model1)
    #model1=Flatten()(model1)
    model1 = Dense(32, activation='relu')(model1)
    model1=Dense(4, activation='softmax')(model1)
    
    
    second_inp=Input(shape=(126,1), name='md_2')
    #second_inp=tf.reshape(second_inp,[21,6,1])
    model2 = Sequential()(second_inp)
    model2=Conv1D(filters=1, kernel_size=100, #padding='same',
                 input_shape=(126,1))(model2)
    model2=MaxPooling1D(pool_size=20)(model2)
    model2 = Dense(4, activation='relu')(model2)
    
    
    #model.compile(loss='binary_crossentropy', optimizer='adam', 
    #             metrics=['binary_accuracy'])    
    #model.add(Embedding(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, input_length = len(X_train[0]), name='embedding_layer'))
    
    #model = Sequential()
    model3=concatenate([model1,model2],axis=1)
    #model3 = Flatten()(model3)
    #model3=Dense(64, activation='relu')(model3)
    
    model3 = Bidirectional(LSTM(units=100, return_sequences=True))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Bidirectional(LSTM(layer_num))(model3)
    model3 = Dropout(dropout)(model3)
    
    model3 = Attention()([model3,model3])
    #model3 = Flatten()(model3)
    #model3 = Dense(16, activation='sigmoid')(model3)
    #model3 = Dense(8, activation='relu')(model3)
    #model3 = Dense(8, activation='relu')(model3)
    
    model3 = Dense(2, activation='sigmoid')(model3)
    #model3 = Dense(2, activation='tanh')(model3)
    ### 最后一层如果是relu，效果会很不好。
    #   tanh还可以，但是不算最好，可以增加epoch后看看
    #   sigmoid也还算可以
    #   droupout从0.8降到0.2后 tpr上升
    #   gamma从3降到2后，tpr再上升
    #   droupout又升到0.6，tpr下降
    #   把relu全部去掉，tpr再次下降
    # 总结，可以考虑先relu，最后tanh+sigmoid，并降低droupout和gamma，并考虑增加正样本量
    
    #model3.add(Attention())
    model=Model(inputs=[first_inp, second_inp], outputs=model3)
    
    '''
    model.compile(optimizer='adam'#tf.keras.optimizers.Adam(learning_rate=0.00001)
                  , loss=#'binary_crossentropy'
                        BinaryFocalLoss(
                                            #from_logits=True
                                            #, apply_class_balancing=True
                                            #,
                                            gamma=gamm
                                            ,pos_weight=pos_alpha
                                            #,alpha=0.9
                                            )
                  #, loss='binary_crossentropy'
                   ,metrics=['accuracy']
                
                )
    
    '''
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=100)
#%%
def create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(35, 4))
                 )
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(35, 4))
                 )
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dense(16, activation='relu'))
    #model.add(Dense(2, activation='softmax'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', 
    #             metrics=['binary_accuracy'])    
    #model.add(Embedding(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, input_length = len(X_train[0]), name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    

    
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
#%%

#%%
def model_run(epoch,serialize_dir,X_train, y_train,model,batch_size):
    param = model.summary()
    print(param)
    history = model.fit(X_train, y_train
                    , batch_size = batch_size
                    , epochs=epoch
                    , verbose=1
                    , validation_split=0.1
                    )
    '''
    model_json = model.to_json()
    with open(serialize_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    '''
    # serialize weights to HDF5
    model.save_weights(serialize_dir+"model.h5")
    print("Saved model to disk")
    return history


#%%
def create_plots(model,trained_model,outpath):
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath+'accuracy.png')
    plt.show()
    plt.clf()

    plt.plot(trained_model.history['loss'])
    plt.plot(trained_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath+'loss.png')
    plt.show()
    plt.clf()

    plot_model(model,to_file=outpath+'model.png')   

#%%
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
def learning_curve(pos_num,neg_num,file):
    datadir = '/home/yangx/DeepSELEX/train_data/'
    bc_onehot,bc_labels,bc_shape = data_process(datadir+file,label_pos=float(pos_num),label_neg=float(neg_num))
    # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.    
    #X_test2=tf.reshape(X_test2,[X_test2.shape[0],21,6]).numpy()
    resultdir='/home/yangx/DeepSELEX/script/Classification/mike/'
    X_train, X_test, y_train, y_test1 = train_test_split(bc_onehot[:], bc_labels[:], test_size=0.1, random_state=42)
    X_train2, X_test2, y_train, y_test = train_test_split(bc_shape[:], bc_labels[:], test_size=0.1, random_state=42)
    list =[]


    BATCH_SIZE = 128
    model_combine= create_combine(gamm=2,pos_alpha=0.25,dropout=0.1)

    #X_train, X_test, = np.expand_dims(X_train, -1).astype("float32"),np.expand_dims(X_test, -1).astype("float32")
    #X_train2=tf.reshape(X_train2,[X_train2.shape[0],21,6]).numpy()
    #combine_history = model_run(resultdir, X_train, y_train, model_combine,BATCH_SIZE)

    combine_history = model_run(100,resultdir, [X_train,X_train2], y_train, model_combine,BATCH_SIZE)
    create_plots(model_combine,combine_history,resultdir)
    y_test, y_score,fpr,tpr,roc_auc=evaluate(model_combine,[X_test,X_test2],y_test,resultdir,'combine')
    return float(roc_auc)
#%%



'''
from sklearn.model_selection import KFold
# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)
acc_per_fold=[]
loss_per_fold=[]
roc = []
# K-fold Cross Validation model evaluation
fold_no = 1
inputs = [bc_onehot,bc_shape][0]
targets=bc_labels
for train, test in kfold.split(inputs,targets):


  
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')
  model_bilstm = create_bilstm()
  # Fit data to model
  history = model_bilstm.fit(inputs[train], targets[train],
              batch_size=500,
              epochs=10,
              verbose=1)
  
  # Generate generalization metrics
  #scores = model_bilstm.evaluate(inputs[test], targets[test], verbose=0)
  y_test, y_score,fpr,tpr,roc_auc=evaluate(model_bilstm,inputs[test], targets[test],resultdir,'bilstm')
  #print(f'Score for fold {fold_no}: {model_bilstm.metrics_names[0]} of {scores[0]}; {model_bilstm.metrics_names[1]} of {scores[1]*100}%')
  #acc_per_fold.append(scores[1] * 100)
  #loss_per_fold.append(scores[0])
  print('Score for fold: ',fold_no,roc_auc)
  roc.append(roc_auc)
  # Increase fold number
  fold_no = fold_no + 1
'''
#%%
def ks(target,resultdir,y_predicted1, y_true1, dnn_type1, y_predicted2, y_true2, dnn_type2, y_predicted3, y_true3, dnn_type3):
  Font={'size':18, 'family':'Arial'}
  
  label1=y_true1
  label2=y_true2
  label3=y_true3
  fpr1,tpr1,thres1 = roc_curve(label1.T[1], y_predicted1.T[1],pos_label=1)
  fpr2,tpr2,thres2 = roc_curve(label2.T[1], y_predicted2.T[1],pos_label=1)
  fpr3,tpr3,thres3 = roc_curve(label3.T[1], y_predicted3.T[1],pos_label=1)
  roc_auc1 = metrics.auc(fpr1, tpr1)
  roc_auc2 = metrics.auc(fpr2, tpr2)
  roc_auc3 = metrics.auc(fpr3, tpr3)
  
  plt.figure(figsize=(6,6))
  plt.plot(fpr1, tpr1, 'b', label = dnn_type1+' = %0.3f' % roc_auc1, color=plt.cm.Paired(0),lw=2)
  plt.plot(fpr2, tpr2, 'b', label = dnn_type2+' = %0.3f' % roc_auc2, color=plt.cm.Paired(1),lw=2)
  plt.plot(fpr3, tpr3, 'b', label = dnn_type3+' = %0.3f' % roc_auc3, color=plt.cm.Paired(2),lw=2)
  plt.legend(loc = 'lower right', prop=Font)
  plt.plot([0, 1], [0, 1],'k--')
  plt.xlim([0, 1.05])
  plt.ylim([0, 1.05])
  plt.ylabel('True Positive Rate', Font,weight='bold')
  plt.xlabel('False Positive Rate', Font,weight='bold')
  plt.tick_params(labelsize=15)
  plt.title(target,Font,weight='bold')
  plt.savefig(resultdir+'roc_auc.png')
  plt.show()
  #return abs(fpr1 - tpr1).max(),abs(fpr2 - tpr2).max(),abs(fpr3 - tpr3).max()
  #reture thres1,thres2,thres3
  return roc_auc1,roc_auc2,roc_auc3

def ks_pr(target,resultdir,y_predicted1, y_true1, dnn_type1, y_predicted2, y_true2, dnn_type2, y_predicted3, y_true3, dnn_type3):
  Font={'size':18, 'family':'Arial'}
  
  label1=y_true1
  label2=y_true2
  label3=y_true3
  precision1,recall1,thres1 = pr_curve(label1.T[1], y_predicted1.T[1])#,pos_label=1)
  precision2,recall2,thres2 = pr_curve(label2.T[1], y_predicted2.T[1])#,pos_label=1)
  precision3,recall3,thres3 = pr_curve(label3.T[1], y_predicted3.T[1])#,pos_label=1)
  pr_auc1 = aps(label1.T[1], y_predicted1.T[1])#,pos_label=1)
  pr_auc2 = aps(label2.T[1], y_predicted2.T[1])#,pos_label=1)
  pr_auc3 = aps(label3.T[1], y_predicted3.T[1])#,pos_label=1)
  
  plt.figure(figsize=(6,6))
  plt.plot(recall1, precision1, 'b', label = dnn_type1+' = %0.3f' % pr_auc1, color=plt.cm.Paired(0),lw=2)
  plt.plot(recall2, precision2, 'b', label = dnn_type2+' = %0.3f' % pr_auc2, color=plt.cm.Paired(1),lw=2)
  plt.plot(recall3, precision3, 'b', label = dnn_type3+' = %0.3f' % pr_auc3, color=plt.cm.Paired(2),lw=2)
  plt.legend(loc = 'lower right', prop=Font)
  plt.plot([0, 1], [0, 1],'k--')
  plt.xlim([0, 1.05])
  plt.ylim([0, 1.05])
  plt.ylabel('Precision', Font,weight='bold')
  plt.xlabel('Recall', Font,weight='bold')
  plt.tick_params(labelsize=15)
  plt.title(target,Font,weight='bold')
  plt.savefig(resultdir+'pr_auc.png')
  plt.show()
  #return abs(fpr1 - tpr1).max(),abs(fpr2 - tpr2).max(),abs(fpr3 - tpr3).max()
  #reture thres1,thres2,thres3

  return [pr_auc1,pr_auc2,pr_auc3], [precision1,precision2,precision3],[recall1,recall2,recall3]


# %%
def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad
def compute_salient_bases(model, x):
    input_tensors = [model.input]
    gradients = model.optimizer.get_gradients(model.output, model.input)
    gradients = model.optimizer.get_gradients(model.output[0][1], [model.input])
    compute_gradients = K.function(inputs = input_tensors, outputs = gradients)
  
    x_value = np.expand_dims(x, axis=0)
    gradients = compute_gradients([x_value])[0][0]
    sal = np.clip(np.sum(np.multiply(gradients,x), axis=1),a_min=0, a_max=None)
    return sal
def interpret(seqdir,data,input_feature,resultdir):
    seq = pd.read_csv(seqdir+data,header=None,sep="\t")
    sequence_index =74000  # You can change this to compute the gradient for a different example. But if so, change the coloring below as well.
    sal = compute_salient_bases(model, input_feature[sequence_index])

    plt.figure(figsize=[30,5])
    barlist = plt.bar(np.arange(len(sal)), sal)
    [barlist[i].set_color('C2') for i in range(20,30)]  # Change the coloring here if you change the sequence index.
    plt.xlabel('Bases')
    plt.ylabel('Magnitude of saliency values')
    plt.xticks(np.arange(len(sal)), list(list(seq[0])[sequence_index]))
    plt.title('Saliency map for bases in one of the positive sequences'
          ' (green indicates the actual bases in motif)')
    
    plt.savefig(resultdir+"interpret.png")
    plt.show()


    
#%%
'''
if __name__ == "__main__":
    #inputfile,seqfile,pos_num,neg_num,outputpath = main(sys.argv[1:])
    datadir = "/home/yangx/DeepSELEX/train_data/"
    inputfile,seqfile,pos_num,neg_num,outputpath = datadir+'CT_20.txt',datadir+'CT-20.txt','7500','150000','./CTGF/7K5/'
    input_features, input_labels= data_process(inputfile,label_pos=float(pos_num),label_neg=float(neg_num))
    X_train, X_test, y_train, y_test = train_test_split(input_features, input_labels, test_size=0.1, random_state=42)

    resultdir = outputpath
    #seqdir = './'
    ''''''
    import tensorflow as tf
    with tf.compat.v1.Session() as sess:
        # To Do
        if "cnn_bilstm" in outputpath:
            model = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
            history = model_run(outputpath,model,500)
        elif "bilstm" in outputpath:
            model = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
            history = model_run(outputpath,model,500)
        elif "cnn" in outputpath:
            model = create_cnn()
            history = model_run(outputpath,model,500)

    create_plots(model, history, outputpath)
    ''''''
    #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
resultdir = outputpath+'cnn/'
model = create_cnn()
#model.load_weights(resultdir+'cnn/model.h5')
history = model_run(resultdir,X_train,y_train,model,500)
y_label_cnn, y_score_cnn, fpr_cnn, tpr_cnnn = evaluate(model,X_test,y_test,resultdir,'CNN')

model = create_bilstm(model,rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
#model.load_weights(resultdir+'bilstm/model.h5')
y_label_lstm, y_score_lstm, fpr_lstm, tpr_lstm = evaluate(X_test,y_test,resultdir,'BiLSTM')    
 
model = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
#model.load_weights(resultdir+'cnn_bilstm/model.h5')
y_label_lstm, y_score_lstm, fpr_lstm, tpr_lstm = evaluate(model,X_test,y_test,resultdir,'CNN-BiLSTM')    
'''
# %%
def output_deepapta(test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath):
    #inputfile,seqfile,pos_num,neg_num,outputpath = main(sys.argv[1:])
    #datadir = "/Volumes/Data Backup/YX/aptamer/train_data/"
    #inputfile,seqfile,pos_num,neg_num,outputpath = datadir+'BC_6.txt',datadir+'BC-6.txt','2000','150000','./BCMA/2k/'
    input_features, input_labels,input_shape= data_process(inputfile,label_pos=float(pos_num),label_neg=float(neg_num))
    X_train, X_test, y_train, y_test = train_test_split(input_features, input_labels, test_size=test_ratio, random_state=42)

    resultdir = outputpath
    #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
    model_cnn = create_cnn()
    model_cnn.load_weights(resultdir+'cnn/model.h5')
    #history_cnn = model_run(resultdir+'cnn/', X_train, y_train, model_cnn,BATCH_SIZE)
    y_label_cnn, y_score_cnn, fpr_cnn, tpr_cnn, roc_auc_cnn = evaluate(model_cnn,X_test,y_test,resultdir+'cnn/','CNN')

    model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
    model_bilstm.load_weights(resultdir+'bilstm/model.h5')
    #history_bilstm = model_run(resultdir+'bilstm/', X_train, y_train, model_bilstm,BATCH_SIZE)
    y_label_lstm, y_score_lstm, fpr_lstm, tpr_lstm, roc_auc_lstm = evaluate(model_bilstm,X_test,y_test,resultdir+'bilstm/','BiLSTM')    
 
    model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
    model_cnnbilstm.load_weights(resultdir+'cnn_bilstm/model.h5')
    #history_cnnbilstm = model_run(resultdir+'cnn_bilstm/', X_train, y_train, model_cnnbilstm,BATCH_SIZE)
    y_label_cnnlstm, y_score_cnnlstm, fpr_cnnlstm, tpr_cnnlstm, roc_auc_cnnlstm = evaluate(model_cnnbilstm,X_test,y_test,resultdir+'cnn_bilstm/','CNN-BiLSTM')
   
   
    ks(resultdir,y_score_cnn,y_label_cnn,"CNN",y_score_lstm,y_label_lstm,"BiLSTM",y_score_cnnlstm,y_label_cnnlstm,"CNN-BiLSTM")  
    ks_pr(resultdir,y_score_cnn,y_label_cnn,"CNN",y_score_lstm,y_label_lstm,"BiLSTM",y_score_cnnlstm,y_label_cnnlstm,"CNN-BiLSTM")  

    return y_score_cnn,y_label_cnn,y_score_lstm,y_label_lstm,y_score_cnnlstm,y_label_cnnlstm
    

    #return model_cnn,model_bilstm,model_cnnbilstm 
  
#%%
class deepapta:
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
        self.onehot_features=np.concatenate((self.onehot_feature[:pos],self.onehot_feature[neg:]))
        self.input_labels=np.concatenate((self.input_label[:pos],self.input_label[neg:]))
        self.shape_features=np.concatenate((self.shape_feature[:pos],self.shape_feature[neg:]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
    #%%
    
    def model(self,BATCH_SIZE,epochs,layer_num):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        #BATCH_SIZE = 32
        self.resultdir = self.outputpath
        #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
        
        self.model_cnn = create_cnn()
        #model_cnn.load_weights(self.resultdir+'cnn/model.h5')
        self.history_cnn = model_run(epochs,self.resultdir+'cnn/', self.X_train, self.y_train, self.model_cnn,BATCH_SIZE)
        create_plots(self.model_cnn,self.history_cnn,self.resultdir+'cnn/')
        self.y_label_cnn, self.y_score_cnn, self.fpr_cnn, self.tpr_cnn, self.roc_auc_cnn = evaluate(self.model_cnn,self.X_test,self.y_test,self.resultdir+'cnn/','CNN')
        
        
        self.model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, dropout = DROPOUT_RATIO)
        #model_bilstm.load_weights(self.resultdir+'bilstm/model.h5')
        self.history_bilstm = model_run(epochs,self.resultdir+'bilstm/'
                                        , self.X_train
                                        #, self.X_train2.reshape(self.X_train2.shape[0],self.X_train2.shape[1],1)
                                        , self.y_train, self.model_bilstm,BATCH_SIZE)
        create_plots(self.model_bilstm,self.history_bilstm,self.resultdir+'bilstm/')
        self.y_label_lstm, self.y_score_lstm, self.fpr_lstm, self.tpr_lstm, self.roc_auc_lstm = evaluate(self.model_bilstm
                                                                 ,self.X_test
                                                                #,self.X_test2.reshape(self.X_test2.shape[0],self.X_test2.shape[1],1)
                                                              ,self.y_test,self.resultdir+'bilstm/','BiLSTM')    
        
        
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
    def model_combine(self,BATCH_SIZE,epochs,layer_num):
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_combine(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
    def model_cnnbilstm_attention(self,BATCH_SIZE,epochs,layer_num):
        self.resultdir = self.outputpath
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
    #%%
    def model_load(self,layer_num,modeldir):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        #BATCH_SIZE = 32
        self.resultdir = self.outputpath
        #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
        
        self.model_cnn = create_cnn()
        self.model_cnn.load_weights(self.resultdir+modeldir+'cnn/model.h5')
        self.y_label_cnn, self.y_score_cnn, self.fpr_cnn, self.tpr_cnn, self.roc_auc_cnn = evaluate(self.model_cnn,self.X_test,self.y_test,self.resultdir+'cnn/','CNN')
        
        
        self.model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, dropout = DROPOUT_RATIO)
        self.model_bilstm.load_weights(self.resultdir+modeldir+'bilstm/model.h5')
        self.y_label_lstm, self.y_score_lstm, self.fpr_lstm, self.tpr_lstm, self.roc_auc_lstm = evaluate(self.model_bilstm
                                                                 ,self.X_test
                                                                #,self.X_test2.reshape(self.X_test2.shape[0],self.X_test2.shape[1],1)
                                                              ,self.y_test,self.resultdir+'bilstm/','BiLSTM')    
        
      
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
      
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.model_cnnbilstm.load_weights(self.resultdir+modeldir+'cnn_bilstm/model.h5')
        #create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')

        #history_cnnbilstm = model_run(100,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, model_cnnbilstm,BATCH_SIZE)
        #y_label_cnnlstm, y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, roc_auc_cnnlstm = evaluate(model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
        
        #self.y_score_cnn,self.y_label_cnn,self.y_score_lstm,self.y_label_lstm,self.y_score_cnnlstm,self.y_label_cnnlstm=y_score_cnn,y_label_cnn,y_score_lstm,y_label_lstm,y_score_cnnlstm,y_label_cnnlstm
        #self.model_cnn,self.model_bilstm,self.model_cnnbilstm=model_cnn,model_bilstm,model_cnnbilstm
    
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
def output_deepselex(test_ratio,inputfile,pos_num,neg_num,outputpath):
    
    input_features, input_labels, input_shape= data_process(features=inputfile,label_pos=pos_num,label_neg=neg_num)
    X_train, X_test, y_train, y_test = train_test_split(input_features, input_labels, test_size=test_ratio, random_state=42)
    X_train = X_train
    y_train = y_train
    resultdir = outputpath
   
    model_cnn = create_deepselex()
    #model_cnn.load_weights(resultdir+'deepselex/model.h5')
    history = model_deepselex_run(resultdir+'deepselex/model.h5', X_train, y_train, resultdir,model_cnn,640)
    y_label_cnn, y_score_cnn, fpr_cnn, tpr_cnn,cm = evaluate(model_cnn,X_test,y_test,resultdir+'deepselex/','DeepSELEX')
    return y_label_cnn, y_score_cnn,fpr_cnn,tpr_cnn#,fpr_lstm,tpr_lstm,fpr_cnnlstm,tpr_cnnlstm
#datadir = "/home/yangx/DeepSELEX/train_data/"
#CT_label_deepse,CT_score_deepse,CT_fpr_deepse,CT_tpr_deepse=output_deepselex(0.1,datadir+'CT_20.txt','7500','150000',weightdir+'7k5/')
#DK_label_deepse,DK_score_deepse,DK_fpr_deepse,DK_tpr_deepse=output_deepselex(0.1,datadir+'DK_30.txt','40000','150000',weightdir+'4k/')
#BC_label_deepse,BC_score_deepse,BC_fpr_deepse,BC_tpr_deepse=output_deepselex(0.1,datadir+'BC_6.txt','2000','150000',weightdir+'2k/')

#%%
import tensorflow as tf
def convolution(input_data, num_input_channels, num_filters, filter_shape, conv_weights,bias_weights,wd1,bd1,W,b,pooling,neuType,training,dropprob):

    model = Sequential()
    # setup the convolutional layer operation
    model = model.add(Conv1d(input_data, conv_weights, 1, padding='VALID'))

    out_layer= model.add().subtract(out_layer,conv_bias)

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform pooling
    if pooling == 'max_pool':
        pool=tf.reduce_max(out_layer,axis=1) 
        
    elif pooling == 'avg_pool':
        out_layer1= tf.reduce_max(out_layer, axis=1)
        out_layer2= tf.reduce_mean(out_layer, axis=1)
        
        x_expanded = tf.expand_dims(out_layer1, 2)                 
        y_expanded = tf.expand_dims(out_layer2, 2)  
        
        concatted = tf.concat([x_expanded, y_expanded], 2)  

        pool = tf.reshape(concatted, [-1, 2*num_filters]) 
        

    t =tf.constant(1 ,dtype=tf.float32)
    
    def ifTrain(pool):
        pooldrop = tf.nn.dropout(pool,keep_prob=dropprob)
#         pooldrop=tf.multiply(pool,mask) 
        out = tf.matmul(pooldrop, wd1) + bd1
        
        return out
    def ifTest(pool):
        out = dropprob*tf.matmul(pool, wd1) + bd1
        return out
    
    #check if there's hidden stage
    if(neuType=='nohidden'):

        out = tf.cond(tf.equal(training,t), lambda: ifTrain(pool), lambda: ifTest(pool))

        
    elif(neuType=='hidden'):


        dense_layer1 = tf.matmul(pool, W) + b
        dense_layer1=tf.nn.relu(dense_layer1)
        
        out = tf.cond(tf.equal(training,t), lambda: ifTrain(dense_layer1), lambda: ifTest(dense_layer1))
        

    return out
    
#%%
from sklearn.svm import SVC
#from pycaret.classification import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
def svc(test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath,dnn_type):
    
    Exp = pd.read_csv(inputfile,header=None,sep='\t')
    data = Exp[Exp.columns[2:]]
    data = pd.concat([data[:int(pos_num)],data[-int(neg_num):]])
    data['affinity'] = [1]*int(pos_num)+[0]*int(neg_num)
    
    clf2 = setup(data = data
                #data.iloc[:,200:]
            , target = 'affinity'
            #,log_experiment = True
            , experiment_name = 'Affinity'
            , session_id = 42
            #,log_plots = True
            #,log_profile = True
            #,log_data = True
            , use_gpu=True
            ,n_jobs=-1
            , silent=True
            , train_size=1-test_ratio
            )
    '''
    model_rf = create_model("rbfsvm")
    plot_model(model_rf,plot='auc')
    plot_model(model_rf,plot='confusion_matrix')
    save_model(model_rf,outputpath+'svm/model')
    '''
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.iloc[:,:-1]), np.array(data['affinity']), test_size=test_ratio, random_state=42)
    model_svm = load_model(outputpath+'svm/model')
    pred_holdout = predict_model(model_svm)
    y_test = get_config('y_test')
    y_score = pred_holdout['Score']
    #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
    '''
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.iloc[:,:-1]), np.array(data['affinity']), test_size=test_ratio, random_state=42)
    clf = SVC(kernel = "rbf"
              ,gamma="auto"
              ,degree = 1
              ,cache_size = 5000
             ).fit(X_train, y_train)
    #score = clf.predict(X_test)
    #svc_fpr, svc_tpr, thresholds = roc_curve(float(y_test),float(score),pos_label=1)
    y_score = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_score)
    print('Confusion matrix:\n',cm)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
    
    plt.savefig(outputpath+'confusion_matrix.png')
    plt.show()
    plt.clf()
    
    Font={'size':18, 'family':'Times New Roman'}

    '''
    Font={'size':18, 'family':'Arial'}
    fpr, tpr, thersholds = roc_curve(y_test, y_score, pos_label=1)
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
    plt.tick_params(labelsize=15)
    plt.savefig(outputpath+'ROC.png')
    plt.show()
    return y_test, y_score,fpr,tpr
    #return fpr, tpr

    

#datadir = "/home/yangx/DeepSELEX/train_data/"
##CT_y_test, CT_y_score,CT_fpr_svm,CT_tpr_svm=svc(0.1,datadir+'CT_20.txt',datadir+'CT-20.txt','7500','150000',weightdir+'7k5/','SVM')
#DK_y_test, DK_y_score,DK_fpr_svm,DK_tpr_svm=svc(0.1,datadir+'DK_30.txt',datadir+'DK-30.txt','4000','150000',weightdir+'4k/','SVM')
#BC_y_test, BC_y_score,BC_fpr_svm,BC_tpr_svm=svc(0.1,datadir+'BC_6.txt',datadir+'BC-6.txt','2000','150000',weightdir+'2k/','SVM')
#

#%%

#CT_label_deepse,CT_score_deepse,CT_fpr_deepse,CT_tpr_deepse=output_deepselex(0.1,datadir+'CT_20.txt',datadir+'CT-20.txt','7500','150000','./7k5/')
##DK_label_deepse,DK_score_deepse,DK_fpr_deepse,DK_tpr_deepse=output_deepselex(0.1,datadir+'DK_30.txt',datadir+'DK-30.txt','4000','150000','./4k/')
#BC_label_deepse,BC_score_deepse,BC_fpr_deepse,BC_tpr_deepse=output_deepselex(0.1,datadir+'BC_6.txt',datadir+'BC-6.txt','2000','150000','./2k/')

# %%
from tensorflow.keras.utils import plot_model
from matplotlib import rcParams
import matplotlib as mpl
sns.set_theme(style="white",font='Arial',font_scale=1.4)
#custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#sns.set_theme(style="ticks", rc=custom_params)
mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'cm' 
mpl.rcParams["axes.linewidth"] = 2
#font = {'family':'Arial','size':45}
#mpl.rc('font',**font)
#mpl.rc('legend',**{'fontsize':45})
mpl.rcParams['savefig.bbox'] = 'tight'
font = {'family' : 'Arial','weight' : 'bold'}  
plt.rc('font', **font) 
def ks_all(dir,title,fpr_cnn,tpr_cnn,fpr_lstm,tpr_lstm,fpr_cnnlstm,tpr_cnnlstm
        , fpr_deepbind, tpr_deepbind
        , fpr_deepselex,tpr_deepselex 
        , fpr_svm,tpr_svm
        ):
    Font={'size':16, 'family':'Arial'}
 
    roc_auc1 = metrics.auc(fpr_cnn,tpr_cnn)
    roc_auc2 = metrics.auc(fpr_lstm,tpr_lstm)
    roc_auc3 = metrics.auc(fpr_cnnlstm,tpr_cnnlstm)
    roc_auc4 = metrics.auc(fpr_deepselex,tpr_deepselex )

    roc_auc5 = metrics.auc(fpr_svm,tpr_svm)
    roc_auc6 = metrics.auc(fpr_deepbind,tpr_deepbind)
  
    plt.figure(figsize=(6,6))
    plt.plot(fpr_cnn,tpr_cnn, color = plt.cm.Paired(0), label = "DeepApta_CNN"+' = %0.3f' % roc_auc1,lw=2)#, color='--r')
    plt.plot(fpr_lstm,tpr_lstm, color = plt.cm.Paired(1), label = "DeepApta_BiLSTM"+' = %0.3f' % roc_auc2,lw=2)#, color='W')
    plt.plot(fpr_cnnlstm,tpr_cnnlstm, color = plt.cm.Paired(2), label = "DeepApta_CNN-BiLSTM"+' = %0.3f' % roc_auc3,lw=2)#, color='K')
    plt.legend(loc = 'lower right', prop=Font, bbox_to_anchor=(1.0,0.01))
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font,weight='bold')
    plt.xlabel('False Positive Rate', Font,weight='bold')
    plt.tick_params(labelsize=15)
    plt.title(title,Font,weight='bold')
    plt.grid(False)
    plt.savefig(dir+'./DeepApta_roc_auc.svg',format='svg',dpi=2500)
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr_svm,tpr_svm,color = plt.cm.Paired(0), label = "SVM"+' = %0.3f' % roc_auc5,lw=2)#, color='K')
    plt.plot(fpr_deepbind,tpr_deepbind,color = plt.cm.Paired(1), label = "DeepBind"+' = %0.3f' % roc_auc6,lw=2)#, color='C')
    plt.plot(fpr_deepselex,tpr_deepselex,color = plt.cm.Paired(2), label = "DeepSELEX"+' = %0.3f' % roc_auc4,lw=2)#, color='C')
    plt.plot(fpr_cnnlstm,tpr_cnnlstm,color = plt.cm.Paired(3), label = "DeepApta"+' = %0.3f' % roc_auc3,lw=2)#, color='K')
    plt.legend(loc = 'lower right', prop=Font, bbox_to_anchor=(1.0,0.01))
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font,weight='bold')
    plt.xlabel('False Positive Rate', Font,weight='bold')
    plt.tick_params(labelsize=15)
    plt.title(title,Font,weight='bold')
    plt.grid(False)
    plt.savefig(dir+'./comparsion_roc_auc.svg',format='svg',dpi=2500)
    plt.show()
#%%   
'''

CT_deepbind=pd.read_csv('/home/yangx/DeepSELEX/Deepbind/CT.csv')
CT_test_deepbind,CT_score_deepbind=CT_deepbind['label'],CT_deepbind['predict_label']
CT_fpr_deepbind, CT_tpr_deepbind, thersholds = roc_curve(CT_test_deepbind,CT_score_deepbind, pos_label=1)
ks_all(weightdir+'7k5/','CTGF',CT_fpr_cnn,CT_tpr_cnn,CT_fpr_lstm,CT_tpr_lstm,CT_fpr_cnnlstm,CT_tpr_cnnlstm
       ,CT_fpr_deepbind, CT_tpr_deepbind
       ,CT_fpr_deepse,CT_tpr_deepse
       ,CT_fpr_svm,CT_tpr_svm
       )



#%%
DK_deepbind=pd.read_csv('/home/yangx/DeepSELEX/Deepbind/DK.csv')
DK_test_deepbind,DK_score_deepbind=DK_deepbind['label'],DK_deepbind['predict_label']
DK_fpr_deepbind, DK_tpr_deepbind, thersholds = roc_curve(DK_test_deepbind,DK_score_deepbind, pos_label=1)
ks_all(weightdir+'4k/','DKK1',DK_fpr_cnn,DK_tpr_cnn,DK_fpr_lstm,DK_tpr_lstm,DK_fpr_cnnlstm,DK_tpr_cnnlstm
       ,DK_fpr_deepbind,DK_tpr_deepbind
       ,DK_fpr_deepse,DK_tpr_deepse
       ,DK_fpr_svm,DK_tpr_svm
        )
#%%
BC_deepbind=pd.read_csv('/home/yangx/DeepSELEX/Deepbind/BC.csv')
BC_test_deepbind,BC_score_deepbind=BC_deepbind['label'],BC_deepbind['predict_label']
BC_fpr_deepbind, BC_tpr_deepbind, thersholds = roc_curve(BC_test_deepbind,BC_score_deepbind, pos_label=1)
ks_all(weightdir+'2k/','BCMA',BC_fpr_cnn,BC_tpr_cnn,BC_fpr_lstm,BC_tpr_lstm,BC_fpr_cnnlstm,BC_tpr_cnnlstm
       ,BC_fpr_deepbind,BC_tpr_deepbind
       ,BC_fpr_deepse,BC_tpr_deepse
       ,BC_fpr_svm,BC_tpr_svm
       )
# %%
def predict_data_process(features,label_pos,label_neg):
    se = []
    #seq = pd.read_csv(datadir+"sequence.txt",header=None)
    with open(features) as sequences:
        for line in sequences:
            line = re.split("\t|\n",line)
            input = [float(x) for x in line[2:142]]
            input_file = []
            for i in range(0,140,4):
                input_file.append(input[i:i+4])
            se.append(input_file)
    label_pos = int(label_pos)
    label_neg = int(label_neg)

    input_features = np.array(se[int(label_pos):int(label_neg)])

    return input_features
def interpret(seqdir,data,inputfile,resultdir,sequence_index,pos_num,neg_num):
    seq = pd.read_csv(seqdir+data,header=None,sep="\t")
    #sequence_index =21  # You can change this to compute the gradient for a different example. But if so, change the coloring below as well.
    input_feature = predict_data_process(inputfile,label_pos=float(pos_num),label_neg=float(neg_num))

    df_cnn = []
    df_cnn = pd.DataFrame(df_cnn)
    model_cnn = create_cnn()
    model_cnn.load_weights(resultdir+'cnn/model.h5')

    df_bilstm = []
    df_bilstm = pd.DataFrame(df_bilstm)
    model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
    model_bilstm.load_weights(resultdir+'bilstm/model.h5')
    
    df_cnnbilstm = []
    df_cnnbilstm = pd.DataFrame(df_cnnbilstm)
    model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
    model_cnnbilstm.load_weights(resultdir+'cnn_bilstm/model.h5')

    for i in range(100):
        df_cnn[i] = compute_salient_bases(model_cnn, input_feature[i])
        df_bilstm[i] = compute_salient_bases(model_bilstm, input_feature[i])
        df_cnnbilstm[i] = compute_salient_bases(model_cnnbilstm, input_feature[i])
    df_cnn.to_csv(resultdir+"cnn/site_importance.csv")
    df_bilstm.to_csv(resultdir+"bilstm/site_importance.csv")
    df_cnnbilstm.to_csv(resultdir+"cnn_bilstm/site_importance.csv")
    
    f, ax= plt.subplots(figsize = (14, 12))
    sns.heatmap(df_cnnbilstm
            ,cmap='rainbow'
            #palettable.cmocean.diverging.Curl_10.mpl_colors
            #, linewidths = 0.08
            , ax = ax
            #, annot=True
            , vmax=0.1
            )
    # 设置Axes的标题
    ax.set_title('saliency of CNN-BiLSTM')
    plt.show()
    plt.close()
    f.savefig(resultdir+'saliency_CNN_BiLSTM.jpg', dpi=1200, bbox_inches='tight')
    
    return df_cnn,df_bilstm,df_cnnbilstm
#%%

datadir = "/home/yangx/DeepSELEX/train_data/"
resultdir = '/home/yangx/DeepSELEX/script/Classification/'
sal_DK_cnn,sal_DK_bilstm,sal_DK_cnnbilstm = interpret(datadir,"DK-30.txt",datadir+'DK_30.txt',resultdir+'4k/',1, '4000', '-150000')
# %%
sal_BC_cnn,sal_BC_bilstm,sal_BC_cnnbilstm = interpret(datadir,"BC-6.txt",datadir+'BC_6.txt',resultdir+'2k/',1, '2000', '-150000')
# %%
sal_CT_cnn,sal_CT_bilstm,sal_CT_cnnbilstm = interpret(datadir,"CT-20.txt",datadir+'CT_20.txt',resultdir+'7k5/',1, '7500', '-150000')
'''