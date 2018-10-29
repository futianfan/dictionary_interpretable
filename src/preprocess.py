import numpy as np 
np.random.seed(1)
f = lambda x:[int(i) for i in x.split()]  ### PLUS 1 
### 1 - 1864, 0 is the padding

def read_train_data(filename):
	lines = open(filename, 'r').readlines()
	lines = lines[1:]  ### delete first line 
	label = [line.split('\t')[0] for line in lines]
	ff = lambda x:1 if x=='True' else 0
	label = list(map(ff,label))
	lines = [line.split('\t')[2] for line in lines]
	lines = list(map(f,lines))
	return lines, label

def embedding2dic(filename):
	lines = open(filename, 'r').readlines()
	lines = lines[1:]
	f = lambda line:np.array([float(i) for i in line.split()[1:]])
	dic = {line.split()[0]:f(line) for line in lines}
	return dic

def data2array(data_dict, MAX_LENGTH, id2vec):
    leng = len(data_dict)
    for k,v in id2vec.items():
    	INPUT_SIZE, = v.shape
    	break 
    arr = np.zeros((leng, MAX_LENGTH, INPUT_SIZE), dtype = np.float32)
    arr_len = []
    for i in range(leng):
        line = data_dict[i]
        line = line[-MAX_LENGTH:]
        for j in range(len(line)):
            try:
                arr[i,j,:] = id2vec[str(line[j])]
            except:
            	##print(line[j])
                pass
        arr_len.append(len(line))
    return arr, arr_len

if __name__ == '__main__':
	lines, label = read_train_data('data/training_data_1.txt')
	print(label[:4])
	EmbedFile = 'data/training_model_by_word2vec_1.vector'
	embeddic = embedding2dic(EmbedFile)
	arr, _ = data2array(lines[:10], 50, embeddic)
	print(arr.shape)

