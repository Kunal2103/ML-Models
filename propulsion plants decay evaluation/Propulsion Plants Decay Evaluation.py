#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load libraries

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#load the data
df = pd.read_csv('C:/python/propulsion.csv')


# ### Exploratory Data Analysis

# In[3]:


#look at the data
df.head(10)


# In[4]:


#shape of the data
df.shape


# In[5]:


#columns
df.columns


# In[6]:


#univariate analysis
df.describe()


# In[7]:


#datatypes
df.dtypes


# ## Data Pre-Processing
# ### Removing unwanted features

# In[8]:


df['GT Compressor inlet air pressure (P1) [bar]'].value_counts()


# * Here, only one value is repeating across all the observations so , it is not carrying variance to explain target variable

# In[9]:


df['GT Compressor inlet air temperature (T1) [C]'].value_counts()


# * similarly, this feature is not useful as it contains only one category

# In[10]:


df['Starboard Propeller Torque (Ts) [kN]'].equals(df['Port Propeller Torque (Tp) [kN]'])


# * Here, this two features have exactly same values, so better to use one feature as both features contains same values
# 
# ### dropping irrelevant features

# In[11]:


df.drop(['Starboard Propeller Torque (Ts) [kN]','GT Compressor inlet air temperature (T1) [C]'
         ,'GT Compressor inlet air pressure (P1) [bar]','Unnamed: 0'] , axis = 1, inplace= True)


# In[12]:


df.head()


# ## Data Pre-Processing
# ### Missing Value Analysis

# In[13]:


df.isnull().sum()


# * no missing value found

# ## Outliers Analysis
# ### Boxplot

# In[14]:


df.shape


# In[15]:


def plot_feature_boxplot(data, features):
    i = 0
    fig, ax = plt.subplots(5,3,figsize=(10,20))
        
    for feature in features:
        i += 1
        plt.subplot(5,3,i)
        sns.boxplot(data[feature]) 
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=6,pad = -6)
        plt.tick_params(axis='y', labelsize=6)
    plt.show()


# In[16]:


features = df.loc[:,:]
plot_feature_boxplot(df ,features)


# * no outliers present in the dataset

# ### Histograms

# In[17]:


def distplot(df , features):
    i = 0
    fig, ax = plt.subplots(5,3,figsize=(15,10))

    
    for feature in features:
        i += 1
        plt.subplot(5,3,i)
        sns.distplot(df[feature]) 
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=6,pad = -6)
        plt.tick_params(axis='y', labelsize=6)
        
    plt.show()
    


# In[18]:


distplot(df,features)


# # 'GT Compressor decay state coefficient.' Prediction

# In[19]:


X = df.drop(['GT Compressor decay state coefficient.', 'GT Turbine decay state coefficient.'],axis = 1)
y = df['GT Compressor decay state coefficient.']


# ### train-test split

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state= 21)

X_train.shape , X_test.shape , y_train.shape , y_test.shape


# ## Model Building
# ### Linear Regression

# In[36]:


#model fitting
lr = LinearRegression().fit(X_train, y_train)


# In[37]:


#prediction
pred_lr  = lr.predict(X_test)


# In[38]:


#accuracy
print('Accuracy =', r2_score(y_test,pred_lr)*100)


# In[39]:


#RMSE
print('RMSE =', np.sqrt(mean_squared_error(y_test,pred_lr)))


# #### Model Evaluation - Linear Regression

# In[41]:


pred_lr_train = lr.predict(X_train)


# In[42]:


print('Accuracy on Train data =',r2_score(y_train, pred_lr_train)*100)


# ### Random Forest

# In[43]:


#model fitting
rf = RandomForestRegressor().fit(X_train, y_train)


# In[44]:


#prediction
pred_rf  = rf.predict(X_test)


# In[45]:


#accuracy
print('Accuracy =', r2_score(y_test,pred_rf)*100)


# In[46]:


#RMSE
print('RMSE =',np.sqrt(mean_squared_error(y_test, pred_rf)))


# * RMSE is very good

# #### Model Evaluation - Random Forest

# In[47]:


pred_rf_train = rf.predict(X_train)


# In[48]:


print('Accuracy on Train data =',r2_score(y_train,pred_rf_train)*100)


# 
# * Random Forest is peforming well

# # 'GT Turbine decay state coefficient.' Prediction

# In[36]:


A = df.drop(['GT Compressor decay state coefficient.', 'GT Turbine decay state coefficient.'],axis = 1)
b = df['GT Turbine decay state coefficient.']


# ### train-test split

# In[37]:


A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.8,test_size=0.2, random_state=21)

A_train.shape , A_test.shape , b_train.shape , b_test.shape


# ## Model Building

# ### Linear Regression

# In[48]:


#model fitting
lr_tb = LinearRegression().fit(A_train, b_train)

#prediction
pred_lr_tb  = lr_tb.predi`ct(A_test)

#accuracy
print('Accuracy =', r2_score(b_test,pred_lr_tb)*100)

#RMSE
print('RMSE =', np.sqrt(mean_squared_error(b_test,pred_lr_tb)))


# ### model evaluation 

# In[52]:


pred_lr_tb_train = lr_tb.predict(A_train)

print('Accuracy on Train data =',r2_score(b_train,pred_lr_tb_train)*100)


# ### Random Forest

# In[49]:


#model fitting
rf_class = RandomForestRegressor().fit(A_train, b_train)

#prediction
pred_rf_tb  = rf_class.predict(A_test)

#accuracy
print('Accuracy =', r2_score(b_test,pred_rf_tb)*100)

#RMSE
print('RMSE =', np.sqrt(mean_squared_error(b_test,pred_rf_tb)))


# ### model evluation

# In[53]:


pred_rf_tb_train = rf_class.predict(A_train)

print('Accuracy on Train data =',r2_score(b_train,pred_rf_tb_train)*100)


# * Random Forest is performing well here as well

# ### Dumping model on local machine

# In[49]:


import pickle
pickle.dump(rf,open('rf_model.pkl','wb'))


# In[ ]:




