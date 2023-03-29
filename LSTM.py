#!/usr/bin/env python
# coding: utf-8

# # **import packages**



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn import metrics
import keras
import tensorflow
from tensorflow.keras.layers import LSTM,Bidirectional

# # **read data**


input_data=pd.read_csv("flow1.csv",header=0)
data=input_data
input_data.head()
data.shape


# # **drop unnecessary columns**



data.drop("Time",axis=1,inplace=True)
data.head()




# # **preprocess data**



data_np=np.array(data)
scaler = MinMaxScaler()
scaler.fit(data_np)
all_data_normalized=scaler.transform(data_np)

#drop label
lbl_normalized=all_data_normalized[:,-1]
data_normalized=all_data_normalized

print("label shape",lbl_normalized.shape)
print("data shape",data_normalized.shape)



# # **create  time series data**



window_size= 30
nfeature=data_normalized.shape[1]

X=[]
y=[]
for i in range(len(data_normalized)-window_size):
    t=[]
    for j in range(0,window_size):
        t.append(data_normalized[[(i+j)], :])
    X.append(t)
    y.append(lbl_normalized[i+ window_size])


data_time_window, lbl_time_window= np.array(X), np.array(y)
data_time_window= data_time_window.reshape(data_time_window.shape[0],window_size, nfeature)
print(data_time_window.shape)
print(lbl_time_window.shape)




# # **split data into train and test sets**



#split data to train and test sets
train_idx= round(.8 * (data_time_window.shape[0]))
train_data=data_time_window[:train_idx,:]
train_lbl=lbl_time_window[:train_idx]
test_data=data_time_window[train_idx:,:]
test_lbl=lbl_time_window[train_idx:]



#strategy = tensorflow.distribute.MirroredStrategy(["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])
#strategy = tensorflow.distribute.MirroredStrategy(["/gpu:0"])



my_epoch=50
my_batch_size=64


# LSTM MODEL



lstm_win_model = keras.models.Sequential()
lstm_win_model.add(keras.layers.LSTM(units=100, input_shape=(data_time_window.shape[1],nfeature)))
lstm_win_model.add(keras.layers.Dense(units=1 ))
lstm_win_model.compile(optimizer="Adam",loss=keras.losses.mean_squared_error,metrics=tensorflow.metrics.MeanAbsoluteError())
lstm_win_model.summary()
#keras.utils.vis_utils.plot_model(lstm_win_model)

#training the model 

lstm_win_model_hist=lstm_win_model.fit(train_data,train_lbl,epochs=my_epoch,batch_size=my_batch_size)


MAE = lstm_win_model_hist.history['mean_absolute_error']
loss = lstm_win_model_hist.history['loss']

plt.figure(figsize=(10,8))
plt.plot(range(my_epoch), loss, label='training loss')
plt.legend(loc='upper right')
plt.title('Training loss ')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('Training_loss_LSTM.png')
plt.show()
plt.figure(figsize=(10,8))
plt.plot(range(my_epoch), MAE, label='training MAE' )
plt.legend(loc='upper right')
plt.title('Training MAE')
plt.savefig('Training_MAE_LSTM.png')
plt.show()



y_pred_lstm_win=lstm_win_model.predict(test_data,verbose=2)
train_pred_lstm_win=lstm_win_model.predict(train_data,verbose=2)
mse=metrics.mean_squared_error(test_lbl, y_pred_lstm_win)
rmse=metrics.mean_squared_error(test_lbl, y_pred_lstm_win,squared=False)
mae= metrics.mean_absolute_error(test_lbl,y_pred_lstm_win)
Rep = open("error_report_LSTM.txt", "w+")
Rep.write('Mean squared error (MSE): %.3f ' % mse)  
Rep.write('Root mean square error (RMSE) : %.3f '%rmse) 
Rep.write('Mean Absolute Error (MAE): %.3f'%mae)
Rep.write('R2 Score: %.9f ' %r2_score(test_lbl,y_pred_lstm_win))
Rep.close()
print('Mean squared error (MSE): %.3f ' % mse)  
print('Root mean square error (RMSE) : %.3f '%rmse) 
print('Mean Absolute Error (MAE): %.3f'%mae)
print("R2 Score",r2_score(test_lbl,y_pred_lstm_win))











