# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:51:24 2022

@author: Damon
"""
from Config import Config
from Core.Monitor_sepsis import PoEMS
from Util.Data_loader import Sepsis_onset,create_dataloader
from Util.util import seed_torch,Normalization,evaluat_matrix,Resample,padding_monitoring_range  
import numpy as np
import copy
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import StratifiedKFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_length(X,l):
    length=[]
    for x in X:        
        length.append((l-len(x[x[:,0]==-1])))
    return length     

def make_forecast_pri(model, loader,alpha):
    model.eval()
    actual, predicted, stopping_point = list(), list(), list()
    for batch_x, batch_y in tqdm(loader['validation']): 
        inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())
        with torch.no_grad():
            y_pred, t = model(inputs,y_true,alpha)                  
            actual.extend(y_true.cpu().detach().numpy())
            predicted.extend(y_pred.cpu().detach().numpy())
            stopping_point.extend(t) 
    actual, predicted, stopping_point = np.array(actual), np.array(predicted), np.array(stopping_point)   
    return actual, predicted, stopping_point
#level=6 #depth of network
#n_channels = [length] * level

#----------------------min max normalizing---------

if Config.Normallize:
    X,scaler=Normalization(X,Config.tool_name)
#-------------------------------------------------------  
def binary_label(y_pre):
    y_pre[y_pre > 0.5] = 1
    y_pre[y_pre <= 0.5] = 0 
    return y_pre

def fit_lstm(model, loader, n_epochs,optimizer, opt_scheduler, sample_tool,beta,alpha,fold,data_name):
    best_auc  = 0.5
    for epoch in range(n_epochs):  
        for phase in ['train','validation']:
            print(f'Epoch {epoch+1}/{n_epochs} ({phase} phase)')
        
            if phase == 'train':
                model.train()  # Set model to training mode 
            elif phase == 'validation':
                model.eval()            
            actual = []
            predicted = []
            stopping_point = []
                                           
            for batch_x, batch_y in tqdm(loader[phase]):
                inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())   
                                        
                optimizer.zero_grad()              
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred, t = model(inputs,y_true,alpha)                    
                    loss, cross_entropy, reward, baseline, penalty = model.applyLoss(y_pred, y_true, beta =beta)                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                actual.extend(y_true.cpu().detach().numpy())
                predicted.extend(y_pred.cpu().detach().numpy())
                stopping_point.extend(t)                             
            # loss and metrics
            actual, predicted, stopping_point = np.array(actual), np.array(predicted), np.array(stopping_point)
            if phase == 'train':
                #print('Train shape',actual.shape,predicted.shape)
                actual,length=actual[:,0],actual[:,1]
                earliness_train= np.mean(stopping_point/length)
                               
                lstm_train_auc = roc_auc_score(actual, predicted)
                predicted=binary_label(predicted)
                lstm_train_acc,train_sen,train_spe,train_pre =evaluat_matrix(confusion_matrix(actual, predicted))      
                print(f'| {phase} auc: {lstm_train_auc:.5f} | acc: {lstm_train_acc:.5f} |Earliness_train: {earliness_train:.5f} | SEN: {train_sen:.5f}')
            elif phase == 'validation':   
                actual,length=actual[:,0],actual[:,1]
                earliness_val= np.mean(stopping_point/length) 
                earliness_val_sepsis=np.mean(stopping_point[actual==1]/length[actual==1])
                
                lstm_val_auc = roc_auc_score(actual, predicted)
                precision, recall, thresholds = precision_recall_curve(actual, predicted)
                lstm_auprc = auc(recall, precision)
                
                predicted=binary_label(predicted)
                lstm_val_acc,val_sen,val_spe,val_pre =evaluat_matrix(confusion_matrix(actual, predicted))
                
                print(f'| {phase} auc: {lstm_val_auc:.5f} |auprc: {lstm_auprc:.5f} |acc: {lstm_val_acc:.5f} |Earliness_val: {earliness_val:.5f} |Earliness_val_sepsis: {earliness_val_sepsis:.5f}|SEN: {val_sen:.5f} |SPE: {val_spe:.5f}')

                #store best model weights
                if lstm_val_auc > best_auc:
                    best_auc = lstm_val_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Save auc model')                            
                torch.save(best_model_wts, 'Weights/Primary/xx.pt')        
                                                                                                                                                       
        opt_scheduler.step()#update every 10 epochs

    model.load_state_dict(best_model_wts)
    
    return model


def main():
    seed_torch(Config.seed)
    if Config.data_name=='6hp_12ho':
       y1=np.ones(2506) #2932
    else:
       y1=np.ones(2932) #2932
    y2=np.zeros(37404)
    y=np.concatenate((y2,y1)) #for minmax, the order is (y2,y1) #
    Y=np.expand_dims(y, axis=1)
    X=np.load('/home/damon/Sepsis/data/minmax/sepsisdata_'+Config.data_name+'.npy',allow_pickle=True)#40336 lentgh fearture
    kf = StratifiedKFold(n_splits=Config.kfold,shuffle=True,random_state=Config.seed) #random split, different each time
    i=0
    AUC,ACC,SEN,SPE,AUPRC,EARLY,EARLY_sepsis=list(),list(),list(),list(),list(),list(),list()
    confuse_matrix=np.zeros([2,2])
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
        x_train[np.isnan(x_train)] =0.5
        x_test[np.isnan(x_test)] = 0.5
          
        if Config.resample:
            y_train=y_train[:,0] #[label]
            x_train , y_train  = Resample(x_train ,y_train,Config.sample_tool)

        length_train=check_length(x_train,24)
        y_train=np.column_stack((y_train, length_train)) #[label,length]
    
        dataloaders = create_dataloader(x_train,y_train,x_test,y_test)  #output tensor data
        
        #build model
        model = PoEMS(x_train.shape[2])
        
        model.Discriminator.load_state_dict(torch.load('/Weights/xx.pt'), strict=False)
        model.BaseRNN.load_state_dict(torch.load('/Weights/xx.pt'), strict=False)
        
        #------------------------------------------------
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.max_lr, weight_decay=Config.weight_decay)
        opt_scheduler = StepLR(optimizer, step_size=Config.step_size, gamma=Config.step_gamma)
        
        # fit model
        model = fit_lstm(model, dataloaders, Config.epoch,optimizer, opt_scheduler,Config.sample_tool,Config.beta,Config.alpha,i,Config.data_name)
        
        # make forecasts
        actual, forecasts, stopping_point = make_forecast_pri(model, dataloaders,Config.alpha)
        actual,length=actual[:,0],actual[:,1]
        print('shape',stopping_point.shape,length.shape)
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
        forecasts=binary_label(forecasts) 
        #-----Confusion_matrix------
        lstm_confuse = confusion_matrix(actual, forecasts)
        confuse_matrix=confuse_matrix+lstm_confuse
        print(f'|AUC: {lstm_auc:.5f}| AUPRC:{lstm_auprc:.5f}| Early:{earliness:.5f} ')
        print(lstm_confuse)
        #-------acc-------sen---------spe
        acc,sen,spe,pre = evaluat_matrix(lstm_confuse)
        ACC.append(acc)
        print('Accuracy : ', acc )
        SEN.append(sen)
        print('Sensitivity : ', sen )
        SPE.append(spe)
        print('Specificity : ', spe)         
        print("The %dth Model_training Done" %i)
         
    print('average_auc:',AUC,sum(AUC) / len(AUC))
    print('average_auprc:',AUPRC,sum(AUPRC) / len(AUPRC))    
    print('average_acc:',ACC,sum(ACC) / len(ACC))
    print('average_sen:',SEN,sum(SEN) / len(SEN))  
    print('average_spe:',SPE,sum(SPE) / len(SPE)) 
    print('average_earliness:',EARLY,sum(EARLY) / len(EARLY))  
    print('average_earliness_sepsis:',EARLY_sepsis,sum(EARLY_sepsis) / len(EARLY_sepsis))   
    print(confuse_matrix) 
    
if __name__ == "__main__":
    main()








