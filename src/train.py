from __future__ import print_function
import math
import torch
import random
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
from model.model import RLP 
from preprocess import read_train_data, embedding2dic, data2array
from sklearn.metrics import roc_auc_score
from test import test_X ##(nnet, data_dict, data_label, epoch, BATCH_SIZE, MAX_LENGTH, embeddic):
from time import time

np.random.seed(1)
torch.manual_seed(7)
SeqFile = 'data/training_data_1.txt'
TestFile = 'data/test_data_1.txt'
EmbedFile = 'data/training_model_by_word2vec_1.vector'
lr = 2e-1
epoch = 50
dropout = 0.5
max_length = 25


lines, label = read_train_data(SeqFile)
test_line, test_label = read_train_data(TestFile)
embeddic = embedding2dic(EmbedFile)
## arr, leng = data2array(lines[:10], 50, embeddic)

for k,v in embeddic.items():
	INPUT_SIZE, = v.shape
	break 
HIDDEN_SIZE = 50    ### hidden 30 (converge faster) => 100 (converge slower) 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 50
BATCH_SIZE = 256
batch_size = 256
KERNEL_SIZE = 10
STRIDE = 1
CNN_OUT_SIZE = int((MAX_LENGTH - KERNEL_SIZE)/STRIDE) + 1
OUT_CHANNEL = INPUT_SIZE
MAXPOOL_NUM = 3
INPUT_SIZE_RNN = OUT_CHANNEL
NUM_HIGH_WAY = 1 
eta = 1e-5 ## 1e-5
rlp = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUT_SIZE, KERNEL_SIZE, OUT_CHANNEL, STRIDE, MAXPOOL_NUM, NUM_HIGH_WAY, BATCH_FIRST = True)

optimizer = optim.SGD(rlp.parameters(), lr=lr)  ### Adam

### seqdata: b, seqlen: numpy.array
leng = len(lines)
num_of_iter = int(np.ceil(leng * 1.0 / batch_size))
for it in range(epoch):
	total_loss = 0
	t1 = time()
	for jt in range(num_of_iter):
		## prepare data 
		input_data = lines[jt * batch_size:jt*batch_size+batch_size] ## numpy array 
		batch_num = len(input_data)
		input_data, leng = data2array(input_data, MAX_LENGTH, embeddic)
		input_label = label[jt*batch_size:jt*batch_size+batch_size]
		input_label = Variable(torch.from_numpy(np.array(input_label)))

		optimizer.zero_grad()
		output, loss_dic = rlp(input_data, leng)  ## input_data: B,T
		lossform = nn.CrossEntropyLoss() 
		loss = lossform(output, input_label) + eta * loss_dic
		loss.backward()
		optimizer.step()
		total_loss += loss.data[0]
	print(str(time() - t1)[:5] + ' seconds. ', end = ' ')
	print('epoch ' + str(it) + ": loss is " + str(total_loss))
	#test_X(rlp, lines, label, it, batch_size, MAX_LENGTH, embeddic)
	test_X(rlp, test_line, test_label, it, batch_size, MAX_LENGTH, embeddic)










