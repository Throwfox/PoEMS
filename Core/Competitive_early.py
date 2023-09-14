import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_label(y_pre):
    y_change=torch.zeros((y_pre.shape)).cuda()
    y_change[y_pre > 0.5] = 1
    y_change[y_pre <= 0.5] = 0 
    return y_change
def exponentialDecay(N):
    tau = 1
    tmax = 4
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.
class Discriminator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        '''
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Linear(128, output_size),
        )
        '''
        #self.softmax = nn.LogSoftmax(dim=1) #test sofmax
        self.sig = nn.Sigmoid()
    def forward(self, h_t):

        y_hat = self.fc(h_t)
        #print('y_hat', y_hat)
        #haha
        y_hat = self.sig(y_hat)
        #fff
        return y_hat

class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        #b_t = self.relu(b_t)
        return b_t
class BaseRNN(nn.Module):

    def __init__(self,
                 N_FEATURES,
                 HIDDEN_DIM,
                 CELL_TYPE="LSTM",
                 N_LAYERS=1):
        super(BaseRNN, self).__init__()

        # --- Mappings ---
        if CELL_TYPE in ["RNN", "LSTM", "GRU"]:
            self.rnn = getattr(nn, CELL_TYPE)(N_FEATURES,
                                              HIDDEN_DIM,
                                              N_LAYERS)
        else:
            try: 
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[CELL_TYPE]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was
                                 supplied, options are ['LSTM', 'GRU',
                                 'RNN_TANH' or 'RNN_RELU']""")
            
            self.rnn = nn.RNN(N_FEATURES,
                              HIDDEN_DIM,
                              N_LAYERS,
                              nonlinearity=nonlinearity)
        self.tanh = nn.Tanh()
        #print("N_FEATURES:", N_FEATURES)
        #print("HIDDEN_DIM:", HIDDEN_DIM)

    def forward(self, x_t, hidden):
        #print(hidden)
        #print('x_t', x_t, x_t.shape)
        output, h_t = self.rnn(x_t, hidden)
        return output, h_t
class Controller_New(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller_New, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

    def forward(self, x):
        probs = torch.sigmoid(self.fc(x.detach()))
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05]).cuda()  # Explore/exploit
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)
class EARLIEST_NEW(nn.Module):
    """Code for the paper titled: Adaptive-Halting Policy Network for Early Classification
    Paper link: https://dl.acm.org/citation.cfm?id=3330974
    Method at a glance: An RNN is trained to model time series
    with respect to a classification task. A controller network
    decides at each timestep whether or not to generate the
    classification. Once a classification is made, the RNN
    stops processing the time series.
    Parameters
    ----------
    ninp : int
        number of features in the input data.
    nclasses : int
        number of classes in the input labels.
    nhid : int
        number of dimensions in the RNN's hidden states.
    rnn_type : str
        which RNN memory cell to use: {LSTM, GRU, RNN}.
        (if defining your own, leave this alone)
    lam : float32
        earliness weight -- emphasis on earliness.
    nlayers : int
        number of layers in the RNN.
    """
    def __init__(self, ninp, nclasses=1, nhid=50, rnn_type="LSTM",
                 nlayers=1, lam=0.0):
        super(EARLIEST_NEW, self).__init__()

        # --- Hyperparameters ---
        self.ninp = ninp
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.lam = lam
        self.nclasses = nclasses

        # --- Sub-networks ---
        self.Controller = Controller_New(nhid+1, 1).cuda()
        self.BaselineNetwork = BaselineNetwork(nhid+1, 1).cuda()
        if rnn_type == "LSTM":
            self.RNN = torch.nn.LSTM(ninp, nhid).cuda()
        else:
            self.RNN = torch.nn.GRU(ninp, nhid)
        self.out = torch.nn.Linear(nhid, nclasses)
        self.sigmoid = nn.Sigmoid()

        #print(self.ninp, self.rnn_type,self.nhid,self.nlayers,self.nclasses)
        
    def initHidden(self, bsz):
        """Initialize hidden states"""
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.nlayers, bsz, self.nhid).to(device),
                    torch.zeros(self.nlayers, bsz, self.nhid).to(device))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid).to(device)

    def forward(self, X, epoch=0, test=False):
        X = torch.transpose(X, 0, 1)
        """Compute halting points and predictions"""
        #if test: # Model chooses for itself during testing
        self.Controller._epsilon = 0.0
        #else:
        #    self.Controller._epsilon = self._epsilon # explore/exploit trade-off
        T,B, V = X.shape
        baselines = [] # Predicted baselines
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        halt_points = -torch.ones((B, self.nclasses)).cuda()
        hidden = self.initHidden(X.shape[1])
        predictions = torch.zeros((B, self.nclasses), requires_grad=True).cuda()
        all_preds = []

        # --- for each timestep, select a set of actions ---
        for t in range(T):
            # run Base RNN on new data at step t
            RNN_in = X[t].unsqueeze(0)
            
            output, hidden = self.RNN(RNN_in, hidden)

            # predict logits for all elements in the batch
            logits = self.out(output.squeeze())
            logits = self.sigmoid(logits)

            # compute halting probability and sample an action
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1, 1).repeat(1, B, 1).cuda()
            c_in = torch.cat((output, time), dim=2)
            a_t, p_t, w_t = self.Controller(c_in)

            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = torch.where((a_t == 1) & (predictions == 0), logits, predictions)

            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = torch.where((halt_points == -1) & (a_t == 1), time.squeeze(0), halt_points)

            # compute baseline
            b_t = self.BaselineNetwork(torch.cat((output, time), dim=2).detach())

            
            if B == 1:
                actions.append(a_t)
                baselines.append(b_t)
                log_pi.append(p_t)
            else:
                actions.append(a_t.squeeze())
                baselines.append(b_t.squeeze())
                log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == -1).sum() == 0:  # If no negative values, every class has been halted
                break

        # If one element in the batch has not been halting, use its final prediction
        if B == 1:
            logits = torch.where(predictions == 0.0, logits, predictions)
        else:
            logits = torch.where(predictions == 0.0, logits, predictions).squeeze()
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze(0)
        #self.locations = np.array(halt_points + 1)
        self.baselines = torch.stack(baselines).squeeze(1).transpose(0, 1)
        if B == 1:
            self.log_pi = torch.stack(log_pi).squeeze(2).transpose(0, 1)
        else:
            self.log_pi = torch.stack(log_pi).squeeze(1).squeeze(2).transpose(0, 1)

        self.halt_probs = torch.stack(halt_probs).transpose(0, 1).squeeze(2)
        
        self.actions = torch.stack(actions).transpose(0, 1)

        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = torch.zeros_like(self.actions).cuda()
        for b in range(B):
            self.grad_mask[b, :(1 + halt_points[b, 0]).long()] = 1
        
        if B == 1:
            return logits, (1+halt_points).mean()/(T+1)
        else:
            return logits.squeeze(), (halt_points+1)#(1+halt_points).mean()/(T+1)

    def applyLoss(self, logits, Y):
        labels, _ = Y[:,0],Y[:,1]
        # --- compute reward ---
        label_pre=binary_label(logits.detach())
        self.r = (2*(label_pre.float() == labels.float()).float()-1).detach().unsqueeze(1)        
        self.R = self.r * self.grad_mask

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # If you want a discount factor, that goes here!
        # It is used in the original implementation.

        # --- compute losses ---
        self.loss_b = F.mse_loss(b, self.R) # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi*self.adjusted_reward).sum()/self.log_pi.shape[1] # RL loss
        self.loss_c = F.binary_cross_entropy(logits, labels) # Classification loss
        self.wait_penalty = self.halt_probs.sum(1).mean() # Penalize late predictions
        self.lam = torch.tensor([self.lam], dtype=torch.float, requires_grad=False).cuda()
        loss = self.loss_r + self.loss_b + self.loss_c + self.lam*(self.wait_penalty)
        # It can help to add a larger weight to self.loss_c so early training
        # focuses on classification: ... + 10*self.loss_c + ...
        return loss

#time_ratio = 0
class MDDNN(nn.Module):
    
    def __init__(self, input_size, time_dim, freq_dim, num_classes):
        super(MDDNN, self).__init__()
        #global time_ratio 
        #time_ratio = time_dim
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=4,stride=1,padding=2),
            nn.BatchNorm1d(64,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2), 
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16,stride=1,padding=8),
            nn.BatchNorm1d(32,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        
        self.lstm = nn.LSTM(32, 32, batch_first=True, num_layers=2)
                
        self.conv1f = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=4,stride=1,padding=2),
            nn.BatchNorm1d(64,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2), 
        )
        
        self.conv2f = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16,stride=1,padding=8),
            nn.BatchNorm1d(32,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        
        self.lstmf = nn.LSTM(32, 32, batch_first=True, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear((self.time_dim*8 + self.freq_dim*32) , 32),
            nn.Dropout(0.1)
        )
        
        self.output = nn.Sequential(
            nn.Linear(32, num_classes),
            #nn.Softmax(dim=1)
        )
        self.sig=nn.Sigmoid()

    def forward(self, x, xf, time_ratio):
        x=x.transpose(1,2)
        xf=xf.transpose(1,2)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = torch.transpose(x, 1, 2)

        x, (xhn, xcn) = self.lstm(x)
        #print('x', x, x.shape)
        x = x.reshape(x.shape[0],-1)
        #print('x', x, x.shape)
        x = x[:,:(time_ratio*8)]
        #print('x', x, x.shape)
        #ggg
        xf = self.conv1f(xf)
        xf = self.conv2f(xf)
        
        xf = torch.transpose(xf, 1, 2)
        xf, (xfhn, xfcn) = self.lstmf(xf)

        xf = xf.reshape(xf.shape[0],-1)
        #print('xf', xf, xf.shape)

        out = torch.cat((x, xf), 1)
        #print('out', out.shape)
        #gg
        
        out = self.fc(out)

        out = self.output(out)

        out = self.sig(out)
        
        return out

import os
def entropy(p):
    return -(p*torch.log(p+1e-12)).sum(1)

def build_t_index(batchsize, sequencelength):
    # linear increasing t index for time regularization
    """
    t_index
                          0 -> T
    tensor([[ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
    batch   [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            ...,
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.]])
    """
    t_index = torch.ones(batchsize, sequencelength) * torch.arange(sequencelength).type(torch.FloatTensor)
    if torch.cuda.is_available():
        return t_index.cuda()
    else:
        return t_index
def build_yhaty(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape
    logprobabilities=logprobabilities.squeeze(2)
    y_pred=binary_label(logprobabilities).cuda()
    (a,b)=torch.where(y_pred==targets)
    y=torch.zeros((batchsize,seqquencelength)).cuda()
    for i in range(len(a)):
        y[a[i]][b[i]]= y_pred[a[i]][b[i]]
    return y
    #eye = torch.eye(nclasses).type(torch.ByteTensor)
    #if torch.cuda.is_available():
    #    eye = eye.cuda()
    # [b, t, c]
    #targets_one_hot = targets.unsqueeze(2).bool()
    #print('target shape',targets_one_hot.shape)
    #implement the y*\hat{y} part of the loss function    
    #y_haty = torch.masked_select(logprobabilities, targets_one_hot)
    #return y_haty.view(batchsize, seqquencelength).exp()
def loss_early_reward(logprobabilities, pts, targets, alpha=0.5, ptsepsilon = 10, power=1):
    targets=targets[:,0]
    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize, sequencelength=seqquencelength)

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    if ptsepsilon is not None:
        ptsepsilon = ptsepsilon / seqquencelength
        #pts += ptsepsilon

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    pts_ = (pts + ptsepsilon)

    b,t,c = logprobabilities.shape
    #loss_classification = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))
    targets = targets.unsqueeze(-1).repeat(1,seqquencelength).long()


    xentropy = F.nll_loss(logprobabilities.transpose(1, 2).unsqueeze(-1), 
        targets.unsqueeze(-1),reduction='none').squeeze(-1)
    #print(xentropy)
    #print(xentropy.shape)

    #loss_classification = alpha * ((pts_ * xentropy)).sum(1).mean()
    loss_classification = F.binary_cross_entropy(logprobabilities, targets.float())
    #print('loss_classification', loss_classification, loss_classification.shape)
    

    yyhat = build_yhaty(logprobabilities, targets)
    #print('yyhat', yyhat, yyhat.shape)
    earliness_reward = (1-alpha) * ((pts) * (yyhat)**power * (1 - (t_index / seqquencelength))).sum(1).mean()
    #print('earliness_reward', earliness_reward, earliness_reward.shape)
    loss = 10*loss_classification - earliness_reward
    #print('loss',loss, loss.shape)

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        earliness_reward=earliness_reward
    )

    return loss, stats
class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
from abc import ABC, abstractmethod
import torch
from sklearn.base import BaseEstimator
#import numpy as np

class EarlyClassificationModel(ABC,torch.nn.Module, BaseEstimator):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass # return logprobabilities, deltas, pts, budget

    @torch.no_grad()
    def predict(self, logprobabilities, deltas):

        def sample_stop_decision(delta):
            delta[delta != delta] = 0
            #print(delta)
            dist = torch.stack([1 - delta, delta], dim=1)
            #print('dist', torch.stack([1 - delta, delta], dim=1))
            #haha
            return torch.distributions.Categorical(dist).sample().byte()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        limit_stop = 0
        
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(stop_now.shape).byte()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)
            
        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).byte()
        
        #stopped[:,int(sequencelength*0.05):] = 1
        
        #print(stopped,batchsize, sequencelength, nclasses)
        # is only true if stopped for the first time
                
        first_stops = (stopped.cumsum(1) == 1) & stopped.bool()
        
        # time of stopping
        t_stop = first_stops.int().argmax(1)
        
        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return predictions_at_t_stop, t_stop

    def attentionbudget(self,deltas):
        batchsize, sequencelength = deltas.shape

        pts = list()

        initial_budget = torch.ones(batchsize)
        if torch.cuda.is_available():
            initial_budget = initial_budget.cuda()

        budget = [initial_budget]
        for t in range(1, sequencelength):
            pt = deltas[:, t] * budget[-1]
            budget.append(budget[-1] - pt)
            pts.append(pt)

        # last time
        pt = budget[-1]
        budget.append(budget[-1] - pt)
        pts.append(pt)

        return torch.stack(pts, dim=-1), torch.stack(budget, dim=1)

    @abstractmethod
    def save(self, path="model.pth",**kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass #return snapshot

class DualOutputRNN(EarlyClassificationModel):
    def __init__(self, input_dim=25, hidden_dims=256, nclasses=1, num_rnn_layers=2, dropout=0.5, bidirectional=False,
                 use_batchnorm=False, use_attention=False, use_layernorm=False, init_late=True):

        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.d_model = num_rnn_layers*hidden_dims

        if not use_batchnorm and not self.use_layernorm:
            self.in_linear = nn.Linear(input_dim, hidden_dims, bias=True)

        if use_layernorm:
            # perform
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.lstmlayernorm = nn.LayerNorm(hidden_dims)

        self.inpad = nn.ConstantPad1d((3, 0), 0)
        self.inconv = nn.Conv1d(in_channels=input_dim,
                  out_channels=hidden_dims,
                  kernel_size=3)

        self.lstm = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=num_rnn_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.sig=nn.Sigmoid()
        if bidirectional: # if bidirectional we have twice as many hidden dims after lstm encoding...
            hidden_dims = hidden_dims * 2

        if use_attention:
            self.attention = Attention(hidden_dims, attention_type="dot")

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dims)

        self.linear_class = nn.Linear(hidden_dims, nclasses, bias=True)
        self.linear_dec = nn.Linear(hidden_dims, 1, bias=True)

        if init_late:
            torch.nn.init.normal_(self.linear_dec.bias, mean=-2e1, std=1e-1)

    def _logits(self, x, is_train):

        # get sequence lengths from the index of the first padded value
        #lengths = torch.argmax((x[:, 0, :] == SEQUENCE_PADDINGS_VALUE), dim=1)
        #lengths = torch.argmax(x[:, 0, :], dim=1)
        # print(lengths)
        
        # if no padded values insert sequencelength as sequencelength
        # lengths[lengths == 0] = maxsequencelength

        #lengths = torch.tensor(input_length)
        # sort sequences descending to prepare for packing
        #lengths, idxs = lengths.sort(0, descending=True)

        # order x in decreasing seequence lengths
        #x = x[idxs]

        #x = x.transpose(1,2) 
        #if is_train == True:
        #    x = x.transpose(1,2)

        if not self.use_batchnorm and not self.use_layernorm:
            x = self.in_linear(x)

        if self.use_layernorm:
            x = self.inlayernorm(x)

        # b,d,t -> b,t,d
        b, t, d = x.shape

        # pad left
        #x_padded = self.inpad(x.transpose(1,2))
        # conv
        #x = self.inconv(x_padded).transpose(1,2)
        # cut left side of convolved length
        x = x[:, -t:, :]
        
        #packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, last_state_list = self.lstm.forward(x)
        #outputs, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)      
        
        if self.use_layernorm:
            outputs = self.lstmlayernorm(outputs)

        if self.use_batchnorm:
            b,t,d = outputs.shape
            o_ = outputs.view(b, -1, d).permute(0,2,1)
            outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d)

        if self.use_attention:
            h, c = last_state_list

            query = c[-1]

            #query = self.bn_query(query)

            outputs, weights = self.attention(query.unsqueeze(1), outputs)
            #outputs, weights = self.attention(outputs, outputs)

            # repeat outputs to match non-attention model
            outputs = outputs.expand(b,t,d)

        logits = self.linear_class.forward(outputs)
        logits = self.sig(logits)
        deltas = self.linear_dec.forward(outputs)

        deltas = torch.sigmoid(deltas).squeeze(2)

        pts, budget = self.attentionbudget(deltas)

        if self.use_attention:
            pts = weights

        return logits, deltas, pts, budget

    def forward(self,x, is_train):
        logits, deltas, pts, budget = self._logits(x,is_train)
        #logits=torch.squeeze(logits,2)
        #logprobabilities = logits
        # stack the lists to new tensor (b,d,t,h,w)
        return logits, deltas, pts, budget
    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot










