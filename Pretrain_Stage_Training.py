# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:51:24 2022

@author: Damon
"""
from Core.Backbone import baseLSTM,baseGRU
from Util.Data_loader import Sepsis_onset,create_dataloader
from Config import Config
from Util.util import seed_torch,Normalization,evaluat_matrix,Resample  
from Util.util import padding_monitoring_range
import numpy as np
import copy
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,auc,precision_recall_curve,average_precision_score,fbeta_score
from sklearn.model_selection import StratifiedKFold

import os
os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def make_forecast_pre(model, loader):
    model.eval()
    actual, predicted = list(), list()
    for batch_x, batch_y in tqdm(loader['validation']): 
        inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())
        with torch.no_grad():
            y_pred = model(inputs)                  
            actual.extend(y_true.cpu().detach().numpy())
            predicted.extend(y_pred.cpu().detach().numpy())
    actual, predicted = np.array(actual), np.array(predicted)    
    return actual, predicted

def check_length(X,l):
    length=[]
    for x in X:        
        length.append((l-len(x[x[:,-1]==-1])))
    return length

#----------------------min max normalizing---------
#if Config.Normallize:
#    X,scaler=Normalization(X,Config.tool_name)
#-------------------------------------------------------  
def binary_label(y_pre):
    y_pre[y_pre > 0.5] = 1
    y_pre[y_pre <= 0.5] = 0 
    return y_pre

def fit_model(model, loader, n_epochs,optimizer, opt_scheduler):
    best_auc  = 0.5
    for epoch in range(n_epochs):  
        for phase in ['train','validation']:
            print(f'Epoch {epoch+1}/{n_epochs} ({phase} phase)')
        
            if phase == 'train':
                model.train()  # Set model to training mode 
            elif phase == 'validation':
                model.eval()            
            running_loss = 0.0
            actual = []
            predicted = []
                    
            for batch_x, batch_y in tqdm(loader[phase]):
                inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())   
                                        
                optimizer.zero_grad()              
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(inputs)
                    if np.isnan(y_pred.cpu().detach().numpy()).any():
                       print('there is Nan in ypre')
                    loss = F.binary_cross_entropy(y_pred, y_true)                 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                                   
                actual.extend(y_true.cpu().detach().numpy())
                predicted.extend(y_pred.cpu().detach().numpy())
                running_loss += loss.item() # * inputs.size(0)
                             
            # loss and metrics
            actual, predicted = np.array(actual), np.array(predicted)
            if phase == 'train':
                lstm_train_auc = roc_auc_score(actual, predicted)
                predicted=binary_label(predicted)
                lstm_train_acc,train_sen,spe,train_pre =evaluat_matrix(confusion_matrix(actual, predicted))         
                print(f'| {phase} auc: {lstm_train_auc:.5f} | acc: {lstm_train_acc:.5f} | SEN: {train_sen:.5f}')                     
            elif phase == 'validation':   
                lstm_val_auc = roc_auc_score(actual, predicted)
                precision, recall, thresholds = precision_recall_curve(actual, predicted)
                lstm_auprc = auc(recall, precision)
                
                predicted=binary_label(predicted)
                lstm_val_acc,val_sen,val_spe,val_pre =evaluat_matrix(confusion_matrix(actual, predicted))
                
                print(f'| {phase} auc: {lstm_val_auc:.5f} |auprc: {lstm_auprc:.5f} |acc: {lstm_val_acc:.5f} | SEN: {val_sen:.5f}| SPE: {val_spe:.5f}')
                
                #store best model weights
                if lstm_val_auc > best_auc:
                    best_auc = lstm_val_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Save auc model')                           
                torch.save(best_model_wts, 'Weights/Pretrain/xx.pt')  
                                                          
        opt_scheduler.step()#update every 10 epochs

    model.load_state_dict(best_model_wts)
    
    return model


def main():
    kf = StratifiedKFold(n_splits=Config.kfold,shuffle=True,random_state=Config.seed) #random split, different each time    
    confuse_matrix=np.zeros([2,2])
    seed_torch(Config.seed)
    y1=np.ones(2932) #2932
    y2=np.zeros(37404)
    y=np.concatenate((y2,y1)) #for minmax, the order is (y2,y1) #
    Y=np.expand_dims(y, axis=1)
    X=np.load('Data/xx.npy',allow_pickle=True)
    
    model_name = Config.model_name
    AUC,ACC,SEN,SPE,PRE,AUPRC=list(),list(),list(),list(),list() ,list()
    i=0  
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
            
        if Config.resample:
            y_train=y_train[:,0] #[label]
            x_train , y_train  = Resample(x_train ,y_train,Config.sample_tool)

        length_train=check_length(x_train,24)
        y_train=np.column_stack((y_train, length_train)) #[label,length]
        y_train, y_test=y_train[:,0], y_test[:,0] #only for classification     
            
        dataloaders = create_dataloader(x_train,y_train,x_test,y_test)
            
            #backbone 
        if model_name=='baseLSTM':           
            model = baseLSTM(x_train.shape[2]) 
        elif model_name=='baseGRU':
            model = baseGRU(x_train.shape[2])
                    
            
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.max_lr, weight_decay=Config.weight_decay)
        opt_scheduler = StepLR(optimizer, step_size=Config.step_size, gamma=Config.step_gamma)
        
        # fit model
        model = fit_model(model, dataloaders, Config.epoch,optimizer, opt_scheduler)
        
        # make forecasts
        actual, forecasts = make_forecast_pre(model, dataloaders)
        
        #---AUPRC-----------
        precision, recall, thresholds = precision_recall_curve(actual, forecasts)
        lstm_auprc = auc(recall, precision)
        ap=average_precision_score(actual,forecasts)
        AUPRC.append(lstm_auprc)
        #--------------------------------
        
        lstm_auc = roc_auc_score(actual, forecasts)
        AUC.append(lstm_auc)
        forecasts=binary_label(forecasts) 
    
        lstm_acc = accuracy_score(actual, forecasts)            
        confusion_fold = confusion_matrix(actual, forecasts)
        confuse_matrix=confuse_matrix+confusion_fold
        print('Final result',f'|  auc: {lstm_auc:.5f} | acc: {lstm_acc:.5f}')
        acc,sen,spe,pre = evaluat_matrix(confusion_fold)
        ACC.append(acc)
        SEN.append(sen)
        SPE.append(spe)
        PRE.append(pre)
        print("The %dth Model_training Done" %i)
              
        print('average_auc=',AUC)
        print('average_auprc=',AUPRC)    
        print('average_acc=',ACC)
        print('average_sen=',SEN)  
        print('average_spe=',SPE)
        print('average_pre=',PRE)
        print(confuse_matrix) 
    
if __name__ == "__main__":
    main()








