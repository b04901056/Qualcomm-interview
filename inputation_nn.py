import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F 
import time
import numpy as np
assert F
import csv,random
from imblearn.over_sampling import SMOTE 

class Datamanager():
    def __init__(self):
        self.dataset = {}  
        self.ipt = 0
        self.max = 0

    def get_data(self,name,file_name,b_size,args,shuf=True):     
        with open(file_name,newline='') as csvfile:
            rows = csv.reader(csvfile)                                                          ## Read file
            data = []                                                                           ## Store the data from file
            for row in rows:
                data.append(row) 
            data = data[2:]
            data = np.array(data) 
            data = np.delete(data,4,1)

            for i in range(data.shape[1]):           
                if i == 3 : 
                    for j in range(data.shape[0]):                                              ## Transform label of attribute #4 '2' to 1(positive), '1' to 0(negative) 
                        if data[j][i] == '1':
                            data[j][i] = 0
                        elif data[j][i] == '2':
                            data[j][i] = 1
                        else:
                            print(j)
                            print('error target')
                elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE': #                           ## Transform label 'TRUE' to 1, 'Negative' to 0
                    for j in range(data.shape[0]):
                        if data[j][i] == 'TRUE':
                            data[j][i] = 1.0
                        elif data[j][i] == 'FALSE':
                            data[j][i] = 0.0
                        else:
                            print(j,i,data[j][i]) 
                            print('other type') 

            top = [108,119,23,69,178,46,92,115,161]
            newdata = data[:,0].reshape(-1,1)
            for x in top:
                newdata = np.concatenate((newdata,data[:,x].reshape(-1,1)),axis = 1)
            self.ipt = newdata[2,:].astype(np.double)  
            newdata = np.delete(newdata,2,0).astype(np.double)
  
            Y = data[:,5] 
            Y = np.delete(Y,2,0).astype(np.long) 
            self.max = np.amax(Y) 
            Y = Y / self.max  
             

 
            X,Y = torch.from_numpy(newdata).cuda(),torch.from_numpy(Y).cuda()                                   ## Convert numpy array to tensor for Pytorch
            train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:])                            ## Wrap up the input/target tensor into TensorDataset   source: https://pytorch.org/docs/stable/data.html
            self.dataset['train'] = Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)     ## Put the TensorDataset in Dataloader (stored in a dictionary), shuffling the samples    source: https://pytorch.org/docs/stable/data.html
 
    
    def train(self,model,trainloader,epoch):                                            ## Train the model
        model.train()                                                                   ## Set to training mode
        optimizer = torch.optim.Adam(model.parameters())                                ## Use Adam optimizer to optimize all DNN parameters    source: https://pytorch.org/docs/stable/optim.html  
        loss_func = nn.BCELoss()                                                        ## Use binary cross entropoy for model evaluation       source: https://pytorch.org/docs/stable/nn.html
        total_loss = 0                                                                  ## Calculate total loss in a epoch
   
        for batch_index, (x, y) in enumerate(trainloader):                              ## Process a batch of data in each timestep
            x, y= Variable(x).cuda(), Variable(y).cuda()  
            output = model(x)                                                           ## Use present model to forecast the the result 
            loss = loss_func(output,y) 
            optimizer.zero_grad()                                                       ## Set the gradient in the previous time step to zero
            loss.backward()                                                             ## Back propagate    source: https://pytorch.org/docs/stable/optim.html
            optimizer.step()                                                            ## Gradient descent    source: https://pytorch.org/docs/stable/autograd.html
            if batch_index % 4 == 0:                                                    ## Print model status    source: https://pytorch.org/docs/stable/optim.html
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t '.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader)),end='')

            total_loss+= loss.data[0]*len(x)                                            ## Sum up batch loss 
 
        total_loss/= len(trainloader.dataset) 
        print('Total loss: {:.4f}'.format(total_loss))  
        a = torch.from_numpy(self.ipt).cuda() 
        print('predicted result: ',model(Variable(a)).data.cpu().numpy() * self.max )
        return total_loss  
 
 

class DNN(nn.Module):                                                                   ## Set up DNN
    def __init__(self,args):
        super(DNN, self).__init__()
        print(args.unit)
        self.den=nn.ModuleList()  
        for i in range(1,len(args.unit)-1):                                             ## Set up hidden layers
            self.den.append( nn.Sequential(
                nn.Linear(args.unit[i-1], args.unit[i]),                                ## Source: https://pytorch.org/docs/stable/nn.html
                nn.ReLU() 
            )) 
        self.den.append( nn.Sequential(
            nn.Linear(args.unit[-2], args.unit[-1]),
            nn.Sigmoid()
        )) 

    def forward(self, x):                                                               ## Connect layers and activation function
        for i in self.den:
            x = i(x) 
        return x 