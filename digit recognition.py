#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


#test train split


# In[3]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
len(x_train)


# In[4]:


len(x_test)


# In[5]:


x_train[0]


# In[6]:


x_train[0].shape


# In[7]:


y_train[0]


# In[8]:


x_train.shape


# In[9]:


x_train_flattend=x_train.reshape(len(x_train),28*28)
x_test_flattend=x_test.reshape(len(x_test),28*28)


# In[10]:


x_train.shape


# In[11]:


x_train=x_train/255
x_test=x_test/255


# In[12]:


x_test_flattend.shape


# In[13]:


x_train_flattend[0]


# In[14]:


plt.matshow(x_train[0])


# In[15]:


y_train[0]


# In[16]:


x_train.shape


# In[17]:


model=keras.Sequential([keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[18]:


model.fit(x_train_flattend,y_train,epochs=10)


# In[19]:


model.evaluate(x_test_flattend,y_test)


# In[20]:


y_predicted=model.predict(x_test_flattend)
y_predicted[0]


# In[21]:


y_train[:6]


# In[22]:


np.argmax(y_predicted[0])


# In[23]:


plt.matshow(x_test[170])


# In[24]:


y_test[170]

