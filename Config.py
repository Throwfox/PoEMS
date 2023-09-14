# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:02:27 2022

@author: Damon
"""

class Config():
    # model transfer map
    seed: int = 882046
    
    kfold: int=10
      
    model_name: str='baseLSTM' # baseLSTM or baseGRU
    
    alpha=0.05 ##bigger alpha, more probs to output, earlier to output.
    beta=0.001 #control scaled time penalty loss if 0.01, huge wave for loss
    
    epoch: int = 30

    max_lr: float = 0.001 # for Monitor
    
    weight_decay: float = 1e-4
    
    batch_size: int = 128
    
    output_size: int = 1

    hidden_size: int = 256
                
    gpu_device: str= '1'

    lock_sepsis=False
#---------------------------------------------------------------------------    
    resample=True #if True, choose sample tool; if False using focal loss
    sample_tool: str='overrandom'
    
    Normallize=False
    norm_tool: str='minmax'# minmax,robust,standard
#---------------------------------------------------------------------------
    #for StepLR to control the decays of learning rate
    step_size=10
    step_gamma=0.5
    
