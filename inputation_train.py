from inputation_nn import Datamanager,DNN
import sys
import numpy as np
import argparse
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
EPOCH = 500
BATCH_SIZE = 256 
parser = argparse.ArgumentParser(description='setting module parameter.') 	 
parser.add_argument('-train', dest='training_set',type=str,required=True)	## Training set file 
parser.add_argument('-u', dest='unit',type=int,nargs='+',required=True) 	## Set DNN layer size
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager() 															## Create Datamanager object
sys.stdout.flush()
dm.get_data('train',args.training_set,BATCH_SIZE,args,True) 				## Read in training data 
print('finish loading data')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dnn = DNN(args).double().cuda()												## Set up DNN and put it on GPU using cuda()
print(dnn) 																	## Print DNN model structure
 
for epoch in range(1,EPOCH + 1):											## Training and testing
	loss = dm.train(dnn,dm.dataset['train'],epoch)  
	print('-'*50) 