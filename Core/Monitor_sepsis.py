import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from ..Config import Config

def binary_label(y_pre):
    y_change=torch.zeros((y_pre.shape))
    y_change[y_pre > 0.5] = 1
    y_change[y_pre <= 0.5] = 0 
    return y_change
    
class Core_LSTM(nn.Module):
    def __init__(self, n_feature, hidden_size=Config.hidden_size, num_layers=1):
        super(Core_LSTM,self).__init__()
        self.input_size = n_feature
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.LSTM=nn.LSTM(input_size=n_feature,hidden_size=hidden_size, 
                        num_layers=self.num_layers,
                        dropout=0.3,batch_first = True)
        
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_normal_(self.LSTM.weight_hh_l0)
        nn.init.xavier_normal_(self.LSTM.weight_ih_l0)
        self.LSTM.bias_ih_l0.data.fill_(0.0)
        self.LSTM.bias_hh_l0.data.fill_(0.0)
                           
    def forward(self, x,hidden):
        output, hidden_cell = self.LSTM(x,hidden)   
        return output,hidden_cell
    
class Controller(nn.Module):

    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)     
        
    def forward(self, h_t, alpha = 0, eps=1):

        out = self.fc(h_t.detach())
        
        probs = torch.sigmoid(out) # Compute halting-probability
        
        probs = (1-alpha) * probs + alpha * torch.FloatTensor([eps]).cuda() #bigger alpha, more probs to output, earlier to output.
        
        #probs = abs(probs-0.3) #----------------reduce the probs to output
        
        m = Bernoulli(probs=probs) # Define bernoulli distribution parameterized with predicted probability
        
        halt = m.sample() # Sample action
        
        log_pi = m.log_prob(halt) # Compute log probability for optimization
        
        return halt, log_pi, -torch.log(probs), probs

class Discriminator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)

        self.AF = nn.Sequential(
          nn.BatchNorm1d(input_size),
          nn.Linear(input_size, 15),
          nn.ELU(),
          nn.Dropout(0.5),
          nn.Linear(15, output_size),
          nn.Sigmoid()
        )
    def forward(self, h_t):

        y_hat = self.AF(h_t.detach())
                
        return y_hat       
class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        #self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t    


class PoEMS(nn.Module):

    def __init__(self, n_feature, hidden_size = Config.hidden_size,hidden_output_size = 1, output_size = 1):
        super(PoEMS, self).__init__()
        self.INPUT_SIZE = n_feature
        self.hidden_size = hidden_size
        self.hidden_output_size = hidden_output_size
        self.output_size = output_size
        self._EPISOLON = 1
        self._LIMIT = 12
        
        # --- Backbone Networks ---#        
        self.BaseRNN = Core_LSTM(n_feature).cuda()
        # --- Discriminator ---#  
        self.Discriminator = Discriminator(hidden_size, output_size).cuda()                    
        # --- Controlling Agent ---#  
        self.Controller = Controller(hidden_size, output_size).cuda()
                  
        self.BaselineNetwork = BaselineNetwork(hidden_size, output_size).cuda()
        

        
    def init_hidden(self,batch_size,hidden_size):        
        return (torch.zeros(1,batch_size,hidden_size).cuda(),
                torch.zeros(1,batch_size,hidden_size).cuda())   
                           
    def forward(self, X,Y,ALPHA):
        length=Y[:,1]        
        
        tau_list = np.zeros(X.shape[0], dtype=int) # maintain stopping point for each sample in a batch
        state_list = np.zeros(X.shape[0], dtype=int) # record which snippet array already stopped in a batch
        
        log_pi = []
        baselines = []
        halt_probs = []
        
        flag = False
        
        hidden = self.init_hidden(len(X),self.hidden_size)   
             
        for t in range(X.shape[1]):
            slice_input=[]
            for idx, x in enumerate(X):
                slice_input.append(x[tau_list[idx],:])
                input_x = torch.stack(slice_input, dim=0)
            input_x=input_x.unsqueeze(1).cuda() #[batch, 1, feature]                    

            # --- Backbone Network ---
            output, hidden = self.BaseRNN(input_x,hidden)
            
            H_t = hidden[0][-1] #[batch,hidden_size]

            rate = self._LIMIT if t > self._LIMIT else t
            


            # --- Controlling Agent ---
            if self.training:
                #set different alpha can affect the performance
                a_t, p_t, w_t, probs = self.Controller(H_t, ALPHA, self._EPISOLON) 
            else:
                a_t, p_t, w_t, probs = self.Controller(H_t, ALPHA, self._EPISOLON) #_ALPHA*rate
            
            # --- Baseline Network ---
            b_t = self.BaselineNetwork(H_t)
                        
            # --- Discriminator ---
            y_hat = self.Discriminator(H_t) 
            label_pre=binary_label(y_hat.detach())

            log_pi.append(p_t)
            
            halt_probs.append(w_t)
            
            baselines.append(b_t)
            
            for idx, a in enumerate(a_t):
                
                if(a == 0 and tau_list[idx] < length[idx]): #
                    tau_list[idx]+=1
                elif (a==1 and tau_list[idx] < length[idx] and label_pre[idx]==0):
                    tau_list[idx]+=1
                else: 
                    state_list[idx] = 1
                    #record the stopping status of a snippet array in the batch
                    
            if (np.mean(state_list)>=1): break 
            # break condition in training phase
            # when all snippet array are stopped, the program will break
                    
                    
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        
        self.baselines = torch.stack(baselines).squeeze(2).transpose(1, 0)

        self.tau_list = tau_list
                
        return y_hat, tau_list

    def applyLoss(self, y_hat, Y, gamma = 1, beta = 0.01):
        labels,length=Y[:,0].unsqueeze(1),Y[:,1]
        # --- compute reward ---
        label_pre=binary_label(y_hat.detach())
        r = (label_pre == labels).float() #checking if it is correct
        R = torch.from_numpy(np.zeros((self.baselines.shape[0],self.baselines.shape[1]))).float().cuda() #reward matrix[batch,length(max t-stop)]
        
        
        
        for idx in range(r.shape[0]):
             # return 1 if correct and -1 if incorrect
            for i in range(self.tau_list[idx]):
                temp_r = (-torch.log((i+1)/length[idx])*2)*r[idx] + torch.log((i+1)/length[idx])
                R[idx][i]=temp_r                
                   
        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()
        
        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        self.loss_c = F.binary_cross_entropy(y_hat.detach(), labels) # Make accurate predictions
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1).mean()
        self.time_penalty = torch.sum(self.halt_probs, dim=1).mean()
        
        # --- collect all loss terms ---
        loss = self.loss_c + gamma * self.loss_r + self.loss_b + beta * self.time_penalty
        
        return loss, self.loss_c, self.loss_r, self.loss_b, self.time_penalty    
