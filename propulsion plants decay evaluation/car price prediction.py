#!/usr/bin/env python
# coding: utf-8

# In[23]:


#load libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.feature_selection import  chi2


# In[52]:


#load the data
cars = pd.read_csv('C:/python/cars_price.csv')


# ### Exploratory Data Analysis

# In[53]:


#look at the data
cars.head()


# In[54]:


#shape of the data
cars.shape


# In[55]:


#columns
cars.columns


# In[56]:


#univariate analysis
cars.describe()


# In[57]:


#datatypes
cars.dtypes


# ## Multi-Variate Analysis
# ### Pair-Plot

# In[208]:


sns.pairplot(cars.drop(['Unnamed: 0'],axis = 1),size = 2)


# ###### We can not see strong linear relationship between variables , but :
# ~ For lower mileage, the priceUSD is high as compared to low mileage cars
# 
# ~ For higher Volume also priceUSD is comparitively high than low Volume cars
# 
# ~ price of cars suddenly increased after year 2000

# ## Data Pre-Processing
# ### Missing Value Analysis

# In[58]:


cars.isnull().sum()


# ### Imputing missing values

# In[59]:


cars['drive_unit'] = cars['drive_unit'].fillna(cars['drive_unit'].mode()[0])
cars['segment'] = cars['segment'].fillna(cars['segment'].mode()[0])
cars['volume(cm3)'] = cars['volume(cm3)'].fillna(cars['volume(cm3)'].median())


# In[60]:


#checking missing values
cars.isnull().sum()


# No mising value found, imputed all

# # Data Visualization

# # Group By
# ### Barplot to see the priceUSD in  each category for qualitative features
# 
# 

# In[61]:


plt.figure(figsize=(18, 10))

plt.subplot(2,2,1)
sns.barplot(cars.transmission, cars.priceUSD)

plt.subplot(2,2,2)
sns.barplot(cars.segment, cars.priceUSD)

plt.subplot(2,2,3)
sns.barplot(cars.color, cars.priceUSD)

plt.subplot(2,2,4)
sns.barplot(cars.drive_unit, cars.priceUSD)


# As we can see, all the features have good variance
# 
# ~ black and brown colored car have high price
# 
# ~ for auto transmission, price is high and other features as shown above

# # Feature Engineering

# In[62]:


#merging two features [make & model] and setting as index
cars['car_name'] = cars['make'].str.cat(cars['model'], sep =" ") 
cars.drop(['make','model','Unnamed: 0'] , axis = 1,inplace = True)
cars = cars.set_index('car_name')


# In[63]:


#creating new year diff feature
cars['current_year'] = 2020
cars['year_diff_frm_currnt_year'] = cars.current_year - cars.year
cars.drop(['current_year','year'],axis = 1 ,inplace =True)


# In[64]:


cars.head()


# # Outliers Analysis
# ### Boxplot

# In[65]:


def plot_feature_boxplot(df, features):
    i = 0
    fig, ax = plt.subplots(1,3,figsize=(20,5))
        
    for feature in features:
        i += 1
        plt.subplot(1,3,i)
        sns.boxplot(df[feature]) 
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=6, pad=-6)
        plt.tick_params(axis='y', labelsize=6)
    plt.show()


# In[66]:


features = cars[['year_diff_frm_currnt_year','mileage(kilometers)', 'volume(cm3)']]
plot_feature_boxplot(cars,features)


#  * there are some extreme outliers, which may affect on final output.

# ### Distribution plot

# In[67]:


plt.figure(figsize=(15, 15))

plt.subplot(3,3,1)
sns.distplot(cars.year_diff_frm_currnt_year)

plt.subplot(3,3,2)
sns.distplot(cars['mileage(kilometers)'])

plt.subplot(3,3,3)
sns.distplot(cars['volume(cm3)'])


# In[68]:


### assuming year_diff_frm_currnt_year follows gausian distribution

uppper_boundary=cars['year_diff_frm_currnt_year'].mean() + 3* cars['year_diff_frm_currnt_year'].std()
lower_boundary=cars['year_diff_frm_currnt_year'].mean() - 3* cars['year_diff_frm_currnt_year'].std()
print(lower_boundary)
print(uppper_boundary)
print(cars['year_diff_frm_currnt_year'].mean())


# In[69]:


#for volume(cm3) feature (skewed )
IQR = cars['volume(cm3)'].quantile(0.75)-cars['volume(cm3)'].quantile(0.25)


#### Extreme outliers
lower_bridge=cars['volume(cm3)'].quantile(0.25)-(IQR*3)
upper_bridge=cars['volume(cm3)'].quantile(0.75)+(IQR*3)
print(lower_bridge)
print(upper_bridge)


# In[70]:


#for mileage(kilometers) [skewed]
IQR=cars['mileage(kilometers)'].quantile(0.75)-cars['mileage(kilometers)'].quantile(0.25)


#### Extreme outliers
lower_bridge=cars['mileage(kilometers)'].quantile(0.25)-(IQR*3)
upper_bridge=cars['mileage(kilometers)'].quantile(0.75)+(IQR*3)
print(lower_bridge)
print(upper_bridge)


# In[71]:


data=cars.copy()


# * removing outliers will end iup with loosing most of the data.
# * let's impute it by following

# In[72]:


#treating outliers
data.loc[data['year_diff_frm_currnt_year']>  40,'year_diff_frm_currnt_year']= 40
data.loc[data['volume(cm3)']> 4400,'volume(cm3)']= 4400
data.loc[data['mileage(kilometers)']> 824044,'mileage(kilometers)']= 824044


# In[73]:


data.head()


# ## chi-square test

# In[74]:


color_fuel_type_crosstab = pd.crosstab(data['color'], data['fuel_type'], 
                                      margins=True)
color_fuel_type_crosstab.head()


# In[75]:


def check_categorical_dependency(crosstab_table, confidence_interval):
    stat, p, dof, expected = stats.chi2_contingency(crosstab_table)
    print ("Chi-Square Statistic value = {}".format(stat))
    print ("P - Value = {}".format(p))
    alpha = 1.0 - confidence_interval
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    return expected


# In[76]:


exp_table_1 = check_categorical_dependency(color_fuel_type_crosstab, 0.95)


# * color and fuel_type features are dependent as p-value is less than 0.05
# ### dropping fuel_type feature

# In[77]:


data = data.drop(['fuel_type'],axis = 1)


# ### Label-Encoding for converting categories into numbers associated with each categories

# In[80]:


le = LabelEncoder()
data = data.apply(le.fit_transform)


# In[81]:


data.head()


# # Feature Selection
# ### Heatmap - Correlation Analysis
# * If corrleation value is [0.7+ or -0.7+] , that two features have high corrleation. It ranges between 1 to -1 

# In[82]:


plt.figure(figsize = (5,5))        # Size of the figure
sns.heatmap(data[['volume(cm3)','mileage(kilometers)','year_diff_frm_currnt_year','priceUSD']].corr(),annot = True ,cmap="BrBG")
plt.show()


# * [year_diff_frm_currnt_year] highly correlated with priceUSD target feature 
# 
# * no multi-colinearity problem as no strong linear relationship between predictor features

# ### Train-Test Split

# In[83]:


X = data.drop('priceUSD',axis = 1)
y = data.priceUSD


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=21)


# In[85]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# ## Feature Scaling
# 
# * To convert continuous features in same scale so, interpretation power will increase

# In[86]:


#normalization
scaler = MinMaxScaler()

X_train[['year_diff_frm_currnt_year', 'mileage(kilometers)', 'volume(cm3)']] = scaler.fit_transform(X_train[['year_diff_frm_currnt_year', 'mileage(kilometers)', 'volume(cm3)']])

X_train.head()


# In[87]:


X_test[['year_diff_frm_currnt_year', 'mileage(kilometers)', 'volume(cm3)']] = scaler.transform(X_test[['year_diff_frm_currnt_year', 'mileage(kilometers)', 'volume(cm3)']])


# ## Model Development
# 
# ### Random Forest

# In[126]:


#model creation
rf = RandomForestRegressor(max_features = 'sqrt').fit(X_train, y_train)


# In[127]:


#prediction on test data
pred  = rf.predict(X_test)


# In[128]:


pred[0:5]


# ##### Model Validation

# In[129]:


#accuracy
print('Accuracy =', r2_score(y_test,pred)*100)


# In[130]:


print('MSE =',mean_squared_error(y_test, pred))


# In[131]:


print('RMSE =',np.sqrt(mean_squared_error(y_test, pred)))


# ##### Model- Performance on training (Evaluation)

# In[132]:


#prediction on train data
pred_train = rf.predict(X_train)


# In[133]:


#accuracy train data
print('Accuracy on Train data =',r2_score(y_train,pred_train)*100)


# ######### Our Model is performing well on test data as well, no Over-fitting problem #########

# In[ ]:




