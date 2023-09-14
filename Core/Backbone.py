import torch.nn as nn
import torch
from ..Config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class baseLSTM(nn.Module):
    def __init__(self, n_feature, batch_size=Config.batch_size,hidden_size=Config.hidden_size, output_size=Config.output_size, num_layers=1):
        super(baseLSTM,self).__init__()
        self.input_size = n_feature
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers=num_layers
        self.LSTM=nn.LSTM(input_size=n_feature,hidden_size=hidden_size, 
                        num_layers=self.num_layers,
                        dropout=0.3,batch_first = True)
        
        self.AF = nn.Sequential(
          #nn.Tanh(),
          nn.BatchNorm1d(hidden_size),
          nn.Linear(hidden_size, 15),
          #nn.Linear(130, 15),
          nn.ELU(),
          nn.Dropout(0.5),
          nn.Linear(15, output_size),
          nn.Sigmoid()
        )
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_normal_(self.LSTM.weight_hh_l0)
        nn.init.xavier_normal_(self.LSTM.weight_ih_l0)
        self.LSTM.bias_ih_l0.data.fill_(0.0)
        self.LSTM.bias_hh_l0.data.fill_(0.0)
        
    def init_hidden(self,batch_size):        
        return (torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_size).to(device),
                torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_size).to(device))  
                           
    def forward(self, x):

        hidden=self.init_hidden(len(x))
        output, hidden_cell = self.LSTM(x,hidden) #Expected hidden[0] size (2, 75, 130), got (2, 256, 130)
        if self.biFlag:
            out= self.AF(torch.cat((hidden_cell[0][-1],hidden_cell[0][-2]),1))     #[10,260]       
        else:
            out= self.AF(hidden_cell[0][-1]) #[10 x 130]        
        return out
    
class baseGRU(nn.Module):
    def __init__(self, n_feature, batch_size=Config.batch_size,hidden_size=Config.hidden_size, output_size=Config.output_size, num_layers=1):
        super(baseGRU,self).__init__()
        self.input_size = n_feature
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers=num_layers
        self.GRU=nn.GRU(input_size=n_feature,hidden_size=hidden_size, 
                        num_layers=self.num_layers,
                        dropout=0.3,batch_first = True)
        
        self.AF = nn.Sequential(
          #nn.Tanh(),
          nn.BatchNorm1d(hidden_size),
          nn.Linear(hidden_size, 15),
          nn.ELU(),
          nn.Dropout(0.5),
          nn.Linear(15, output_size),
          nn.Sigmoid()
        )
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_normal_(self.GRU.weight_hh_l0)
        self.GRU.bias_hh_l0.data.fill_(0.0)
        
    def init_hidden(self,batch_size):        
        return torch.zeros(1,batch_size,self.hidden_size).to(device) 
                           
    def forward(self, x):

        hidden=self.init_hidden(len(x))
        output, hidden_cell = self.GRU(x,hidden) #Expected hidden[0] size (2, 75, 130), got (2, 256, 130)
        out= self.AF(hidden_cell[-1]) #[10 x 130]        
        return out
