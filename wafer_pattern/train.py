import os
from os.path import join 
import numpy as np
import pandas as pd 
import keras
from keras import layers, Input, models 
from sklearn.model_selection import train_test_split
from keras.models import load_model

epoch = 30
batch_size = 128
shape = [25,27]

x = np.load('X.npy')
label = np.load('Y.npy') 

label_set = {}
label_to_id = {} 
id_to_label = {}

for i in range(label.shape[0]):
    if str(label[i][0]) == 'Edge-Loc':
        label[i][0] = 'Loc' 

for i in range(label.shape[0]):
    if str(label[i][0]) in label_set:
        label_set[str(label[i][0])] += 1
    else:
        label_set[str(label[i][0])] = 1 
print(label_set) 

num_none = int(label_set['none'] * 0.0)
new_x = []
new_label = []

count = 0
for i in range(label.shape[0]):
    if label[i][0] == 'none':
        if count < num_none:
            new_x.append(x[i])
            new_label.append(label[i][0])
            count += 1
    else:
        new_x.append(x[i])
        new_label.append(label[i][0])

new_x = np.array(new_x)
new_label = np.array(new_label).reshape(-1,1)

print(new_x.shape)
print(new_label.shape)

count = 0
for category in label_set:
	label_to_id[str(category)] = int(count) 
	id_to_label[int(count)] = str(category) 
	count += 1
 
for i in range(new_label.shape[0]):
	new_label[i][0] = label_to_id[str(new_label[i][0])] 

y = []
for i in range(new_label.shape[0]):
	tmp = np.zeros(shape = (1,len(label_set)))
	tmp[0][int(new_label[i][0])] = 1
	y.append(tmp)
y = np.array(y).reshape(-1,len(label_set))   

label_set = {}
for i in range(new_label.shape[0]):
    if id_to_label[int(new_label[i][0])] in label_set:
        label_set[id_to_label[int(new_label[i][0])]] += 1
    else:
        label_set[id_to_label[int(new_label[i][0])]] = 1 

print(label_set) 

x_train, x_test, y_train, y_test = train_test_split(new_x.astype('float32'), y.astype('float32'), test_size=0.2)

def create_cnn_model():
    input_shape = (shape[0], shape[1], 3)
    input_tensor = Input(input_shape)

    conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
    conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
    conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)

    flat = layers.Flatten()(conv_3)

    dense_1 = layers.Dense(512, activation='relu')(flat)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    output_tensor = layers.Dense(7, activation='softmax')(dense_2)

    model = models.Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model

cnn = create_cnn_model()

cnn.fit(x_train, y_train,
	   validation_data=[x_test, y_test],
       batch_size=batch_size,
       epochs=epoch,
       verbose=1)

cnn.save('my_model.h5') 
 
cnn = load_model('my_model.h5')
print(cnn.summary())
answer = []
predict = []

print(y_test.shape)

for i in range(len(y_test)): 
    if not np.argmax(y_test[i],axis = 0) == label_to_id['none']:
        answer.append(np.argmax(y_test[i],axis = 0)) 
        predict.append(np.argmax(cnn.predict(np.expand_dims(x_test[i], axis=0)),axis = 1))

count = 0
for i in range(len(answer)):
	print(id_to_label[int(answer[i])] , ' , ' , id_to_label[int(predict[i])])
	if answer[i] == predict[i]:
		count += 1

print('Accu = ',float(count / len(answer)))
