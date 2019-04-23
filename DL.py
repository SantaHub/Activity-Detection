#!/usr/bin/env python3
"""
Created on Sat Dec  9 21:55:23 2017

@author: austin
"""
#Inside Docker deb /home/ActivityDetect
# cd /home/austin/ML/MotionDetection


import matplotlib.pyplot as plt
import numpy as np
import keras


import pandas as pd
all_data = pd.read_csv('data/data.csv')

activities = set(all_data['activity'])

# Print number of instances for each activity
for i in activities:
    print( len ( all_data.loc[all_data['activity']==i] ) )

data=(all_data.loc[all_data['activity']=='ly'][:2880])
data=data.append(all_data.loc[all_data['activity']=='walk'][:2880])
data=data.append(all_data.loc[all_data['activity']=='stand'][:2880])
data=data.append(all_data.loc[all_data['activity']=='sit'][:2880])
data=data.append(all_data.loc[all_data['activity']=='cycl'][:2880])
data=data.append(all_data.loc[all_data['activity']=='b2'][:2880])
data=data.append(all_data.loc[all_data['activity']=='b1'][:2880])

data['activity']=data['activity'].replace(activities,list(range(0,len(activities))))

dataX = data.as_matrix().astype(np.float)
datay = data['activity']
y = np.array(datay)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(dataX) #features made unit varient

def build_model():
    from keras.models import Sequential
    from keras.layers import Dropout, Conv1D, LSTM, ELU, Dense, Activation,GlobalMaxPooling1D, Embedding
    
    
    model=Sequential()
    
    model.add(Dense(16,input_shape=(8,)))
    model.add(Activation("sigmoid"))
    
    model.add(Dense(7))
    model.add(Activation("softmax"))

    return model

## Split data
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

model = build_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
early_release = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')
tbCallback = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)
Result=model.fit(X_train, y_train, batch_size=32, epochs=10,validation_data=(X_test,y_test))

##Plot Classifier compare
plt.plot(Result.history['acc'])
plt.plot(Result.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
# plt.savefig('model_name'+clf_type+'_acc.png')
plt.show()

