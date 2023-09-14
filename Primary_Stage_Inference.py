# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:51:24 2022

@author: Damon
"""
from Config import Config
from Core.Monitor_sepsis import PoEMS
from Util.Data_loader import Sepsis_onset,create_dataloader
from Util.util import seed_torch,Normalization,evaluat_matrix,padding_monitoring_range  
from Primary_Stage_Training import make_forecast_pri

import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,auc,precision_recall_curve,fbeta_score
import torch
from tqdm import tqdm
import os
import statistics
from sklearn.model_selection import StratifiedKFold
os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(Config.seed)

def binary_label(y_pre):
    y_change=torch.zeros((y_pre.shape))
    y_change[y_pre > 0.5] = 1
    y_change[y_pre <= 0.5] = 0 
    return y_change
def check_length(X):
    length=[]
    for x in X:        
        length.append((25-len(x[x[:,0]==-1])))
    return length 

def main():
    #-------------------------load data---------------------------------
    y1=np.ones(2932) #2932
    y2=np.zeros(37404)
    y=np.concatenate((y2,y1)) #for minmax, the order is (y2,y1) #
    Y=np.expand_dims(y, axis=1)
    X=np.load('/Data/processed_sepsis_24h.npy',allow_pickle=True)

    #-----------------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=Config.kfold,shuffle=True,random_state=Config.seed) #random split, different each time
    for alpha in [0,0.3,0.5,0.7,1]:
        i=0
        AUC,ACC,SEN,SPE,AUPRC,PRE,EARLY,EARLY_sepsis,HM = list(),list(),list(),list(),list(),list(),list(),list(),list()
        Fbeta_precision,Fbeta_recall=list(),list()
        confuse_matrix=np.zeros([2,2])
        
        actual_list, forecasts_list, stopping_point_list=list(),list(),list()
        
        for train_ind,test_ind in kf.split(X,Y):
            i+=1
            print('Fold', i, 'out of',10)
            x_train,y_train = X[train_ind], Y[train_ind] #x_train,y_train = X[train_ind,:,:],Y[train_ind]
            x_test, y_test  = X[test_ind],  Y[test_ind]

            x_train,y_train=padding_monitoring_range(x_train,y_train,24)
            x_test, y_test=padding_monitoring_range(x_test, y_test,24)
               
            x_train=np.stack(x_train) #shape from (num,) to (num,length(fixed),features)
            x_test =np.stack(x_test) 
            
            #deal with nan
            x_train[np.isnan(x_train)] =0
            x_test[np.isnan(x_test)] = 0
            
            dataloaders = create_dataloader(x_train,y_train,x_test,y_test)  #output tensor data
            model  = PoEMS(x_test.shape[2])
            
            #-------------load model
            model.load_state_dict(torch.load('Weights/Primary/xx.pt'))#24h
            #------------------------------------------------
            model.to(device)
                            
            # make forecasts
            actual, forecasts, stopping_point = make_forecast_pri(model, dataloaders,alpha) #0.05~1 
  
            actual_list.append(actual)
            forecasts_list.append(forecasts)
            stopping_point_list.append(stopping_point)                       
            actual,length=np.expand_dims(actual[:,0],1),np.expand_dims(actual[:,1],1)
            stopping_point=np.expand_dims(stopping_point,1)

            print('For alpha equal to ' ,alpha,'------------------------------------------')
           #---Earliness-----------            
            earliness= np.mean(stopping_point/length)
            earliness_sepsis=np.mean(stopping_point[actual==1]/length[actual==1])
            EARLY.append(earliness)
            EARLY_sepsis.append(earliness_sepsis)
           #---AUPRC-----------
            precision, recall, thresholds = precision_recall_curve(actual, forecasts)
            lstm_auprc = auc(recall, precision)
            AUPRC.append(lstm_auprc)

           #-----AUC------------------------          
            lstm_auc = roc_auc_score(actual, forecasts)
            AUC.append(lstm_auc)
                   
           #-----Confusion_matrix------
            forecasts=binary_label(forecasts) 
            confusion_fold = confusion_matrix(actual, forecasts)
            confuse_matrix=confuse_matrix+confusion_fold

           #-----------f beta------------
            fbeta_precision = fbeta_score(actual, forecasts, 0.5)
            fbeta_recall = fbeta_score(actual, forecasts, 2)            
            Fbeta_precision.append(fbeta_precision)
            Fbeta_recall.append(fbeta_recall)
          
           #-------acc-------sen---------spe
            acc,sen,spe,pre = evaluat_matrix(confusion_fold)
            ACC.append(acc)
            SEN.append(sen)
            SPE.append(spe)
            PRE.append(pre)
            
            #--------------HM---------------------------
            hm=(2*(1-earliness_sepsis)*acc)/((1-earliness_sepsis)+acc)
            HM.append(hm)
            print("The %dth Model_testing Done" %i)
            print("-------------------------------------------------------------------")
        print('For alpha equal to ' ,alpha,'------------------------------------------')     
        
        print('average_auc:',AUC,statistics.mean(AUC),statistics.stdev(AUC))
        print('average_auprc:',AUPRC,statistics.mean(AUPRC),statistics.stdev(AUPRC))    
        print('average_acc:',ACC,statistics.mean(ACC),statistics.stdev(ACC))
        print('average_sen:',SEN,statistics.mean(SEN),statistics.stdev(SEN))  
        print('average_spe:',SPE,statistics.mean(SPE),statistics.stdev(SPE))
        print('average_pre:',PRE,statistics.mean(PRE),statistics.stdev(PRE))    
        print('average_earliness:',EARLY,statistics.mean(EARLY),statistics.stdev(EARLY)) 
        print('average_earliness_sepsis:',EARLY_sepsis,statistics.mean(EARLY_sepsis),statistics.stdev(EARLY_sepsis))
        print('average_HM:',HM,statistics.mean(HM),statistics.stdev(HM))
        print('average_Fbeta_precision:',Fbeta_precision,statistics.mean(Fbeta_precision),statistics.stdev(Fbeta_precision)) 
        print('average_Fbeta_recall:',Fbeta_recall,statistics.mean(Fbeta_recall),statistics.stdev(Fbeta_recall))             
        print(confuse_matrix)

        #---Utility
        stoplist=np.concatenate(stopping_point_list,axis=0)
        forecast=np.concatenate(forecasts_list,axis=0)
        actual=np.concatenate(actual_list,axis=0)
        forecast=binary_label(forecast)    
        forecast=np.squeeze(forecast)
        actual,length=actual[:,0],actual[:,1]
        early=length-stoplist
        u_total_list=np.zeros_like(actual)
        for i in range(len(forecast)):
            if actual[i]==0 and forecast[i]==1: #for non-sepsis: FP , punishment -0.05,  
                u_total_list[i] = -0.05
            elif actual[i]==1 and forecast[i]==1: #for sepsis: TP, reward
                if early[i] <= 6:
                    u_total_list[i]=1-(6-early[i])*float(1/9) #no more than 6 hours before
                elif early[i] > 6 and early[i]<= 12:
                    u_total_list[i]=(12-early[i])*float(1/6) # 6~ 12 hours before
                elif early[i] > 12:                  # more than 12 hours before
                    u_total_list[i]=-0.05
            elif actual[i]==1 and forecast[i]==0: #for sepsis: FN, punishment
                if early[i] <= 6:
                    u_total_list[i]=-(6-early[i])*float(2/9)
        u_total=np.sum(u_total_list) 
        print(u_total)
        Umin,Umax=-(4/3)*2932,2932
        #Umin only make non-sepsis predictions at the last hour, 
        Unom=(u_total-Umin)/(Umax-Umin)
        print('For alpha:',alpha,'U total:',u_total,'U norm:',Unom)                   
if __name__ == "__main__":
    main()
