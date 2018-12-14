import numpy as np 
import csv,sys  
from scipy.stats.stats import pearsonr 
import matplotlib.pyplot as plt 
import math 

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)                                                          ## Read file
    data = []                                                                           ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:] 
    data = np.array(data) 

    print(pearsonr(data[:,3].astype(np.double),data[:,116].astype(np.double))[0])
    print(pearsonr(data[:,3].astype(np.double),data[:,162].astype(np.double))[0])   
    input()
    data = np.delete(data,176,1)            ## These columns have std = 0 => remove
    data = np.delete(data,171,1)
    data = np.delete(data,170,1)
    data = np.delete(data,169,1)         
    data = np.delete(data,167,1)
    data = np.delete(data,166,1) 
    data = np.delete(data,165,1)  
    data = np.delete(data,5,1)  
    data = np.delete(data,4,1)  

    feature_num = data.shape[1] 
    for i in range(feature_num):           
        if i == 3: 
            for j in range(data.shape[0]):                                              ## Transform label of attribute #4 '2' to 1(positive), '1' to 0(negative) 
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print(j)
                    print('error target')
        elif i == 163:                                                                  ## One hot encoding
            one_hot = []
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    one_hot.append([1,0])
                elif data[j][i] == '3':
                    one_hot.append([0,1])
                else:
                    print(j,i,data[j][i]) 
                    print('other type') 
            one_hot = np.array(one_hot).reshape(-1,2) 
            data = np.concatenate((data,one_hot),axis = 1)                   
        elif i == 164:                                                                  ## One hot encoding
            one_hot = []
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    one_hot.append([1,0,0])
                elif data[j][i] == '3':
                    one_hot.append([0,1,0])
                elif data[j][i] == '9909':
                    one_hot.append([0,0,1])
                else:
                    print(j,i,data[j][i]) 
                    print('other type') 
            one_hot = np.array(one_hot).reshape(-1,3) 
            data = np.concatenate((data,one_hot),axis = 1) 
        elif i == 169:                                                                  ## One hot encoding
            one_hot = []
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    one_hot.append([1,0])
                elif data[j][i] == '99':
                    one_hot.append([0,1])
                else:
                    print(j,i,data[j][i]) 
                    print('other type') 
            one_hot = np.array(one_hot).reshape(-1,2) 
            data = np.concatenate((data,one_hot),axis = 1) 
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':                             ## One hot encoding
            one_hot = []
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    one_hot.append([1,0])
                elif data[j][i] == 'FALSE':
                    one_hot.append([0,1])
                else:
                    print(j,i,data[j][i]) 
                    print('other type') 
            one_hot = np.array(one_hot).reshape(-1,2) 
            data = np.concatenate((data,one_hot),axis = 1)  

    data = np.delete(data,170,1)
    data = np.delete(data,169,1)
    data = np.delete(data,167,1)
    data = np.delete(data,166,1)
    data = np.delete(data,165,1) 
    data = np.delete(data,164,1)
    data = np.delete(data,163,1)
    data = data.astype(np.double) 

    for i in range(data.shape[1]):  
        if i == 3:
            continue 
        mean = data[:,i].astype(np.double).mean()
        std = data[:,i].astype(np.double).std()
        if(std == 0):                                           
            print(name,' ',i,' ',data[0][i])
        data[:,i] = (data[:,i].astype(np.double) - mean) / std 

cor = []

for i in range(data.shape[1]):                                          ## Compute the variables'correlation with target variable 
    c = pearsonr(data[:,3],data[:,i])[0] 
    if not math.isnan(c):
        cor.append([int(i+1),abs(c)]) 

cor = np.array(cor)  
cor = cor[cor[:,1].argsort()] 
print(cor)
print(cor.shape)

cor = cor[-11:-1,:]

x = np.arange(10)                                                       ## Plot the histogram with matplotlib
plt.bar(x, height= cor[:,1])
plt.xticks(x, [str(cor[i][0]) for i in range(cor.shape[0])])
plt.show()

