import numpy as np 
import csv , sys  
from scipy.stats.stats import pearsonr 
import matplotlib.pyplot as plt 
import math 

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)              ## Read training data
    data = []                               ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,2,0)              ## Missing data => remove

    #data = np.delete(data,0,0)              ## Positive(attribute #4 = 2) outlier => remove   
    #data = np.delete(data,4,0) 
    #data = np.delete(data,5,0) 
    #data = np.delete(data,6,0) 
    #data = np.delete(data,11,0) 
    #data = np.delete(data,2542,0)  

    #data = np.delete(data,5,1)
    data = np.delete(data,4,1)

    for i in range(data.shape[1]): 
        if i == 3 :                                                     ## Transform label of attribute #4 '2' to 1(positive), '1' to 0(negative) 
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print('error target')
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':             ## Transform label 'TRUE' to 1, 'Negative' to 0
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    data[j][i] = 1
                elif data[j][i] == 'FALSE':
                    data[j][i] = 0

                else:
                    print(j,i,data[j][i]) 
                    print('other type')   
data = data.astype(np.double)
cor = []

for i in range(data.shape[1]):                                          ## Compute the variables'correlation with target variable 
    c = pearsonr(data[:,3],data[:,i])[0] 
    if not math.isnan(c):
        cor.append([int(i+1),abs(c)])
    #print(i+1,':',c)

cor = np.array(cor)  
cor = cor[cor[:,1].argsort()] 
print(cor)
print(cor.shape)

cor = cor[160:170,:]

x = np.arange(10)                                                       ## Plot the histogram with matplotlib
plt.bar(x, height= cor[:,1])
plt.xticks(x, [str(cor[i][0]) for i in range(cor.shape[0])])
plt.show()

