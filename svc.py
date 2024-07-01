from sklearn.svm import SVC
#from pycaret.classification import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
def model_svc(test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath,dnn_type):
    
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