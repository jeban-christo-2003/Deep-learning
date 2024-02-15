#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv('insurance2.csv')
df.head()


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(df[['age','affordability']],df.bought_insurance,test_size=0.2,random_state=25)


# In[10]:


x_test


# In[11]:


x_train_scaled=x_train.copy()
x_train_scaled['age']=x_train_scaled['age']/100
x_test_scaled=x_test.copy()
x_test_scaled['age']=x_test_scaled['age']/100


# In[12]:


x_test_scaled


# In[ ]:


model = keras.Sequential([keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_scaled,y_train,epochs=5000)


# In[64]:


x_test_scaled


# In[63]:


model.predict(x_test_scaled)

