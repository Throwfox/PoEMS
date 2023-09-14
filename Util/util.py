from imblearn.under_sampling import EditedNearestNeighbours,RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
import torch
import os
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from tqdm import tqdm

def focal_loss(bce_loss, targets, gamma=2, alpha=0.075):
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()

def evaluat_matrix(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    Accuracy= (tp+tn)/(tp+fp+fn+tn)
    Sensitivity= tp/(tp+fn) #equal to recall
    Specificity= tn/(fp+tn)
    Precision = tp/(tp+fp)
    return Accuracy,Sensitivity,Specificity,Precision
def Resample(X,Y,tool):
    X_r = X.reshape(X.shape[0], -1)
    if tool=='smote':
        sampler= SMOTE()
    elif tool=='underenn':
        sampler= EditedNearestNeighbours()
    elif tool=='overrandom':
        sampler= RandomOverSampler()
    elif tool=='underrandom':
        sampler =RandomUnderSampler()    
    X_res, y_res = sampler.fit_sample(X_r, Y)   
    X_res=X_res.reshape(X_res.shape[0],X.shape[1],X.shape[2])
    return X_res,y_res
def seed_torch(r_seed):
    random.seed(r_seed)
    os.environ['PYTHONHASHSEED'] = str(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    torch.cuda.manual_seed(r_seed)
    torch.cuda.manual_seed_all(r_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Normalization(X,tool_name):
    X_r = X.reshape((-1,X.shape[2]), order='F')
    if tool_name=='minmax':
      scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_r) 
    elif tool_name=='robust':
      scaler = RobustScaler().fit(X_r)
    elif tool_name=='standard':
      scaler = StandardScaler().fit(X_r) #mean 0, variance 1      
    X_r = scaler.transform(X_r)   
    X_norm=X_r.reshape((X.shape[0],X.shape[1],X_r.shape[1]), order='F')
    return X_norm,scaler

def padding_monitoring_range(X,Y,t):
    length=[]
    for idx,x in tqdm(enumerate(X)):
       length.append(len(x)) 
       if len(x)<t:
           pad=np.array(np.full([24-len(x),x.shape[1]],-1))#np.nan
           X[idx]=np.concatenate((x,pad),axis=0)
    Y=np.column_stack((Y, length)) #y_train[:,1] length; y_train[:,0] label
    return X,Y   
def window_slice_augmentation(x_train,y_train):
    index_of_one = np.where(y_train == 1)
    X_sepsis=x_train[index_of_one[0]]
    count=0
    augmentation=[] 
    for x in X_sepsis:        
        if len(x)==25:
            count+=1
            for j in range(7,25):
                augmentation.append(x[-j:])
    x_train=np.append(x_train,augmentation)                      
    y1=np.ones(count*18)
    y1=np.expand_dims(y1, axis=1)
    y_train=np.concatenate((y_train, y1), axis=0)            
    return   x_train,y_train 