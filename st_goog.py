#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:36:42 2022

@author: avi_patel
"""
#pip install streamlit
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models,utils
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *

st.markdown("GOOG Stock Prediction, Enter Prior 15 Days' price")
progress = st.progress(0)

features = []
features.append(st.text_input(label = "15 days ago" , value = 1000))
features.append(st.text_input(label = "14 days ago" , value = 1000))
features.append(st.text_input(label = "13 days ago" , value = 1000))
features.append(st.text_input(label = "12 days ago" , value = 1000))
features.append(st.text_input(label = "11 days ago" , value = 1000))
features.append(st.text_input(label = "10 days ago" , value = 1000))
features.append(st.text_input(label = "9 days ago" , value = 1000))
features.append(st.text_input(label = "8 days ago" , value = 1000))
features.append(st.text_input(label = "7 days ago" , value = 1000))
features.append(st.text_input(label = "6 days ago" , value = 1000))
features.append(st.text_input(label = "5 days ago" , value = 1000))
features.append(st.text_input(label = "4 days ago" , value = 1000))
features.append(st.text_input(label = "3 days ago" , value = 1000))
features.append(st.text_input(label = "2 days ago" , value = 1000))
features.append(st.text_input(label = "1 day ago" , value = 1000))



X = pd.DataFrame({
    "1" : [features[0]],
    "2" : [features[1]],
    "3" : [features[2]],
    "4" : [features[3]],
    "5" : [features[4]],
    "6" : [features[5]],
    "7" : [features[6]],
    "8" : [features[7]],
    "9" : [features[8]],
    "10" : [features[9]],
    "11" : [features[10]],
    "12" : [features[11]],
    "13" : [features[12]],
    "14" : [features[13]],
    "15" : [features[14]],
})
progress.progress(100)

#d2=np.array([1000,1002,1005,970,1009,1003,1012,1009,1003,1012,1022,1029,1019,1014,1007])
#d2= features.astype('float32')
#d2 = np.reshape(X, (-1, 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#d2f = scaler.fit_transform(d2)
#d2x= np.reshape(d2f, (d2f.shape[1], 1, d2f.shape[0]))
#d2xhat=model.predict(d2x)
#prediction=scaler.inverse_transform(d2xhat)
model=tf.keras.models.load_model('/Users/avi_patel/Documents/googlstm/')
features=np.array(features,dtype=float)
if st.button("Submit"):
    d2 = np.reshape(features, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    d2f = scaler.fit_transform(d2)
    d2x= np.reshape(d2f, (d2f.shape[1], 1, d2f.shape[0]))
    d2xhat=model.predict(d2x)
    prediction=scaler.inverse_transform(d2xhat)
    
    st.subheader("Next day's predicted price is':")

    st.write(prediction)