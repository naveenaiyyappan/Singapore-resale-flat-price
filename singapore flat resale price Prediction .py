#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


price_1 = pd.read_csv("C:\\Users\\Ivin\\Downloads\\ResaleFlatPricesBasedonApprovalDate19901999.csv")
price_2 = pd.read_csv("C:\\Users\\Ivin\\Downloads\\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
price_3 = pd.read_csv("C:\\Users\\Ivin\\Downloads\\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
price_4 = pd.read_csv("C:\\Users\\Ivin\\Downloads\\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
price_5 = pd.read_csv("C:\\Users\\Ivin\\Downloads\\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")


# In[3]:


price_1.head()


# In[4]:


price_2.head()


# In[5]:


price_3.head()


# In[6]:


price_4.head()


# In[7]:


price_5.head()


# ###### merge the values

# In[8]:


prices = pd.concat([price_1,price_2,price_3], sort = False)


# In[9]:


prices = pd.concat([prices, price_4,price_5],axis = 0, ignore_index=True, sort=False)


# In[10]:


prices["month"] = pd.to_datetime(prices["month"])


# In[11]:


prices["month"]


# In[12]:


prices.info()


# In[13]:


prices.columns


# In[14]:


len(prices)


# In[15]:


prices.shape


# ###### Null values

# In[16]:


prices.isna().sum()


# In[17]:


prices.isna().sum()/len(prices)*100


# In[18]:


prices = prices.dropna()


# In[19]:


prices.isnull().sum()/len(prices)*100


# In[20]:


prices["town"].unique()


# In[21]:


prices["flat_type"].unique()


# In[22]:


prices["block"].unique()


# In[23]:


prices["street_name"].unique()


# In[24]:


prices["storey_range"].unique()


# In[25]:


prices["floor_area_sqm"].unique()                                        


# In[26]:


prices["flat_model"].unique()


# In[27]:


prices["lease_commence_date"].unique()


# In[28]:


prices["resale_price"].unique()


# In[29]:


prices["remaining_lease"].unique()


# In[30]:


prices.info()


# In[31]:


prices.dtypes


# In[40]:


prices.drop_duplicates()


# In[52]:


import statistics


# In[53]:


def get_median(x):
    split_list = x.split(' TO ')
    float_list = [float(i) for i in split_list]
    median = statistics.median(float_list)
    return median

prices['storey_median'] = prices['storey_range'].apply(lambda x: get_median(x))
prices


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[54]:


categorical_cols = [i for i in prices.columns if prices[i].dtype =="object"]


# In[55]:


categorical_cols


# In[56]:


numeric_cols = [ i for i in prices if i not in categorical_cols]


# In[57]:


numeric_cols


# In[58]:


col = ['town','flat_type','block','street_name','storey_range','flat_model','remaining_lease']


# In[59]:


for i in col:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(data=prices)
    plt.title(f'Boxplot of {i}')
    plt.xlabel(i)
    plt.show()


# In[60]:


df = prices


# In[61]:


df['floor_area_sqm'] = np.log(df['floor_area_sqm'])
sns.boxplot(x='floor_area_sqm', data=df)
plt.show()


# In[63]:


df['storey_median'] = np.log(df['storey_median'])
sns.boxplot(x='storey_median', data=df)
plt.show()


# In[64]:


df['resale_price'] = np.log(df['resale_price'])
sns.boxplot(x='resale_price', data=df)
plt.show()


# In[65]:


df.dtypes


# In[66]:


corrMatrix = df.corr()
plt.figure(figsize=(15, 10))
plt.title("Correlation Heatmap")
sns.heatmap(
    corrMatrix, 
    xticklabels=corrMatrix.columns,
    yticklabels=corrMatrix.columns,
    cmap='RdBu', 
    annot=True
)


# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


X=df[["floor_area_sqm","lease_commence_date","storey_median"]]
y=df['resale_price']

# Normalizing the encoded data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[69]:


test_df = pd.DataFrame(X)
test_df


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[72]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[73]:


from sklearn.linear_model import LinearRegression


# In[74]:


LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.score(X_train, y_train))
print(LR.score(X_test, y_test))


# In[75]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


# In[76]:


DTR = DecisionTreeRegressor()
# hyperparameters
param_grid = {
    'max_depth': [2, 5, 10, 15, 20, 22],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    'max_features': ['auto', 'sqrt', 'log2']
}


# In[77]:


# gridsearchcv
grid_search = GridSearchCV(estimator=DTR, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# In[78]:


# evalution metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(" ")
print('Mean squared error:', mse)
print('Mean Absolute Error', mae)
print('Root Mean squared error:', rmse)
print(" ")
print('R-squared:', r2)


# In[82]:


new_sample = np.array([[8740, 999, np.log(44), 55, np.log(11)]])
new_sample = scaler.transform(new_sample[:, :3])
new_pred = best_model.predict(new_sample)[0]
np.exp(new_pred)


# In[83]:


import pickle


# In[84]:


with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

