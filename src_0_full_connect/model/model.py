import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import model 
np.random.seed(1)
torch.manual_seed(7)


class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUT_SIZE, KERNEL_SIZE, OUT_CHANNEL, STRIDE, MAXPOOL_NUM, NUM_HIGH_WAY, BATCH_FIRST = True):
        super(RLP, self).__init__() 
        INPUT_SIZE_RNN = INPUT_SIZE  
        self.INPUT_SIZE = INPUT_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NUM_LAYER = NUM_LAYER
        self.OUT_SIZE = OUT_SIZE
        self.KERNEL_SIZE = KERNEL_SIZE
        self.OUT_CHANNEL = OUT_CHANNEL
        self.STRIDE = STRIDE
        self.MAXPOOL_NUM = MAXPOOL_NUM
        self.NUM_HIGH_WAY = NUM_HIGH_WAY
        self.BATCH_FIRST = BATCH_FIRST
        self.rnn1 = nn.LSTM(
            input_size = INPUT_SIZE_RNN, 
            hidden_size = int(HIDDEN_SIZE / 2),
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True
            )
        self.out1 = nn.Linear(HIDDEN_SIZE, OUT_SIZE)
        self.out2 = nn.Linear(OUT_SIZE, 2)
        self.out3 = nn.Linear(HIDDEN_SIZE, 2)

        ####  CNN 
        self.conv1 = nn.Conv1d(in_channels = INPUT_SIZE, out_channels = OUT_CHANNEL, kernel_size = KERNEL_SIZE, stride = STRIDE)   
        self.maxpool = nn.MaxPool1d(kernel_size = MAXPOOL_NUM)

        #### highway
        self.num_layers = NUM_HIGH_WAY
        self.nonlinear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.linear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.gate = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.f = F.relu 

    def forward_highway(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

    def forward_rnn(self, X_batch, X_len):
        batch_size = X_batch.shape[0]
        #X_batch = Variable(torch.from_numpy(X_batch).float())
        dd1 = sorted(list(range(len(X_len))), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        ### Option I
        #X_out, _ = self.rnn1(pack_X_batch, None) 
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]
        ### Option II
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)
        X_out2 = X_out2[dd]    ## batch_size, HIDDEN_SIZE
        return X_out2

    def forward_A(self, X_batch, X_len):
    	X_batch = torch.from_numpy(X_batch).float()
    	X_batch = Variable(X_batch)
    	batch_size = X_batch.shape[0]

        ### cnn + rnn
    	X_batch_2 = X_batch.permute(0,2,1)  ## batch size, INPUT_SIZE, MAX_LENGTH
    	X_batch_3 = self.conv1(X_batch_2)
    	X_batch_4 = self.maxpool(X_batch_3)
    	f_map = lambda x: max(int((int((x - self.KERNEL_SIZE) / self.STRIDE) + 1) / self.MAXPOOL_NUM),1)
    	X_len2 = list(map(f_map, X_len))
    	X_batch_4 = X_batch_4.permute(0,2,1)
    	X_out2 = self.forward_rnn(X_batch_4, X_len2)
    	return X_out2 

    def forward(self, X_batch, X_len):
    	X_len = list(X_len)
    	X_out2 = self.forward_A(X_batch, X_len)
    	X_out2 = self.forward_highway(X_out2)   ### add highway: 0.685 => 0.676
    	X_out6 = F.softmax(self.out3(X_out2))
    	return X_out6


if __name__ == '__main__':
	INPUT_SIZE = 100 ## embedding size
	HIDDEN_SIZE = 30 
	OUT_SIZE = 30
	NUM_LAYER = 1
	BATCH_FIRST = True 
	MAX_LENGTH = 50
	BATCH_SIZE = 256

	KERNEL_SIZE = 10
	STRIDE = 1
	CNN_OUT_SIZE = int((MAX_LENGTH - KERNEL_SIZE)/STRIDE) + 1
	OUT_CHANNEL = INPUT_SIZE
	MAXPOOL_NUM = 3
	INPUT_SIZE_RNN = OUT_CHANNEL
	NUM_HIGH_WAY = 1 
	nn = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUT_SIZE, KERNEL_SIZE, OUT_CHANNEL, STRIDE, MAXPOOL_NUM, NUM_HIGH_WAY, BATCH_FIRST = True)





