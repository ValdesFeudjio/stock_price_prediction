import time 
import datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
# from keras.src import backend
# from keras.src.backend.tensorflow import *
# from keras.src.backend.tensorflow import core
# import tensorflow as tf
import streamlit as st


ticker = 'AAPL'
period1 = int(time.mktime(datetime.datetime(2010,1,1,23,59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2019,12,31,23,59).timetuple()))
interval = '1d'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{user_input}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(query_string)
print(df)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart')
ma100 =df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart')
ma100 =df.Close.rolling(100).mean()
ma200 =df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

x_train =[]
y_train =[]

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train =np.array(x_train), np.array(y_train)  

model =load_model('keras_model.h5')

past_100_days =data_training.tail(100)

final_df =past_100_days._append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test =np.array(x_test), np.array(y_test)  
y_predicted = model.predict(x_test)
scaler= scaler.scale_


scale_factor =1/scaler[0]
y_predicted =y_predicted*scale_factor
y_test =y_test*scale_factor

st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label ='Original Price')
plt.plot(y_predicted,'r',label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)