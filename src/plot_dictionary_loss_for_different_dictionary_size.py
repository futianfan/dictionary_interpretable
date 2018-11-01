
import matplotlib.pyplot as plt

x = [3, 5, 10, 20, 30] 
y = [70.32, 68.82, 62.83, 61.28, 59.18] ## average
y = [i * 50 for i in y]  ## 50 batches 

plt.plot(x, y, 'b-') 
plt.xlabel('Size of Dictionary')
plt.ylabel('Dictionary Learning Loss')
plt.savefig('./dictionary_learning_loss.png')

