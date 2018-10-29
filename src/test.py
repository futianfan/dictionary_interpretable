import numpy as np 
import torch 
from preprocess import data2array
from sklearn.metrics import roc_auc_score

def test_X(nnet, data_dict, data_label, epoch, BATCH_SIZE, MAX_LENGTH, embeddic):
    N_test = len(data_dict)
    assert len(data_dict) == len(data_label)
    iter_num = int(np.ceil(N_test * 1.0 / BATCH_SIZE))
    y_pred = []
    y_label = []
    for i in range(iter_num):
        stt = i * BATCH_SIZE
        endn = min(N_test, stt + BATCH_SIZE)
        batch_x, batch_len = data2array(data_dict[stt:endn], MAX_LENGTH, embeddic)
        if batch_x.shape[0] == 0:
            break
        output, _ = nnet(batch_x, batch_len)
        output_data = output.data 
        for j in range(output_data.shape[0]):
            y_pred.append(output_data[j][1])
            y_label.append(data_label[stt+j])
    auc = roc_auc_score(y_label, y_pred)
    print('AUC of Epoch ' + str(epoch) + ' is ' + str(auc)[:5])







