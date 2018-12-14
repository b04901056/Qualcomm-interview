import numpy as np 
import csv , sys
from sklearn.decomposition import PCA       ## Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)                                                          ## Read file
    data = []                                                                           ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:] 
    data = np.array(data)     

    data = np.delete(data,0,0)              ## Positive(attribute #4 = 2) outlier => remove 
    data = np.delete(data,4,0) 
    data = np.delete(data,5,0) 
    data = np.delete(data,6,0) 
    data = np.delete(data,11,0) 
    data = np.delete(data,2542,0) 

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

    Y = data[:,3].reshape(-1,1).astype(np.double)                       ## Extract attribute #4 as targets
    X = np.delete(data,3,1).astype(np.double)


    print(X.shape)
    print(Y.shape)

    sm = SMOTE(sampling_strategy = 1)                                  ## Use SMOTE to generate minor class samples    source: https://imbalanced-learn.org/en/stable/generated/imblearn.over_sampling.SMOTE.html
    X, Y = sm.fit_resample(X, Y) 
    ''' 
    count_0 = 0
    count_1 = 0
    count_1_list = []
    for i in range(Y.shape[0]):
        if Y[i][0] == 0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1
            count_1_list.append(i)
    print('count_0:',count_0)
    print('count_1:',count_1)
                                                                         ## Copy the positive (attribute #4 = 2) samples
      
    ori_one_X , ori_one_Y = X[count_1_list] , Y[count_1_list] 
    for i in range(100): 
        noise = np.random.normal(0, 0.3, ori_one_X.shape)
        add_one_X = ori_one_X + noise 
        X = np.concatenate((X,add_one_X),axis = 0)
        Y = np.concatenate((Y,ori_one_Y),axis = 0)
    ''' 
    '''
    number = 1500
    while(number > 0): 
        for i in range(Y.shape[0]):
            if Y[i][0] == 0:
                X = np.delete(X,i,0)
                Y = np.delete(Y,i,0)
                number = number - 1
                break 
    ''' 


    pca = PCA(n_components = 2)                                         ## Use PCA map the data onto a two-dimensional plane
    newData = pca.fit_transform(X)  

    plt.figure(figsize = (8,8))                                         ## Source: https://matplotlib.org/gallery/lines_bars_and_markers/scatter_with_legend.html
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 component PCA') 
	
    Positive = []
    Negative = []

    for i in range(Y.shape[0]):
    	if Y[i] == 1:
    		Positive.append(newData[i])
    	else:
    		Negative.append(newData[i])
 
    Positive = np.array(Positive)
    Negative = np.array(Negative)

    print(Positive.shape)
    print(Negative.shape) 

    plt.scatter(Positive[:,0], Positive[:,1], c = 'r', s = 50 , alpha = 0.5 , label = 'Positive')  
    plt.scatter(Negative[:,0], Negative[:,1], c = 'g', s = 50 , alpha = 0.5 , label = 'Negative')  
    
    plt.legend(loc='upper right')
    plt.show()
