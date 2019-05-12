import os
from os.path import join
import numpy as np
import pandas as pd 
 
df = pd.read_pickle('LSWMD.pkl')

def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort() 
    return [ backitems[i][1] for i in range(0,len(backitems))] 

def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1

df['waferMapDim']=df.waferMap.apply(find_dim)
'''
size_to_count = {}
for i in range(len(df)):
    if len(df.iloc[i,:]['failureType']) == 0:
        continue
    if df.iloc[i,:]['waferMapDim'] in size_to_count:
        size_to_count[df.iloc[i,:]['waferMapDim']] += 1
    else:
        size_to_count[df.iloc[i,:]['waferMapDim']] = 1
print(sort_by_value(size_to_count)) 
'''
shape = [25,27]

sub_df = df.loc[df['waferMapDim'] == (shape[0], shape[1])]
sub_wafer = sub_df['waferMap'].values

sw = []
label = []

for i in range(len(sub_df)):
    # skip null label
    if len(sub_df.iloc[i,:]['failureType']) == 0:
        continue
    sw.append(sub_df.iloc[i,:]['waferMap'].reshape(1, shape[0], shape[1]))
    label.append(sub_df.iloc[i,:]['failureType'][0][0])
 
x = np.array(sw).reshape(-1,shape[0],shape[1])
y = np.array(label).reshape((-1,1))
# check dimension
print('x shape : {}, y shape : {}'.format(x.shape, y.shape))
dic = {}
for i in range(y.shape[0]):
	if y[i][0] not in dic:
		dic[y[i][0]] = 1
	else:
		dic[y[i][0]] += 1
for key in dic:
	print(key,':',str(dic[key]))
  
np.save('Y.npy' , y)

new_x = np.zeros((len(x), shape[0], shape[1], 3))

for w in range(len(x)):
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_x[w, i, j, int(x[w, i, j])] = 1
print(new_x)
input()
np.save('X.npy',new_x)