#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[33]:


df=pd.read_csv(r"C:\Users\Akash\Downloads\archive (3).zip")
df


# In[34]:


df.head()


# In[35]:


df.shape


# In[36]:


df.tail()


# In[37]:


df.info()


# In[38]:


df.isnull().sum()


# In[39]:


df.describe()


# In[40]:


corelation=df.corr()
corelation


# In[41]:


#construct a heatmap to understand corelation
plt.figure(figsize=(8,8))
sns.heatmap(corelation,fmt='.2f',square=True,annot=True,annot_kws={'size':15},cmap='Purples')


# In[42]:


corelation['GLD']


# In[43]:


sns.distplot(df['GLD'])


# In[44]:


X=df.drop(['Date','GLD'],axis=1)
X


# In[45]:


Y=df['GLD']
Y


# In[46]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=8)


# In[47]:


X_train.shape


# In[48]:


Y_train


# In[49]:


x_test.shape


# In[20]:


Y_train.shape


# In[21]:


regressor=RandomForestRegressor(n_estimators=100)


# In[22]:


regressor.fit(X_train,Y_train)#X_train represent all the features whereas Y_train represnt corespondind Gold values to the X_train features


# In[51]:


prediction_on_training_data=regressor.predict(X_train)
print(prediction_on_training_data)


# In[24]:


prediction_on_training_data.shape


# In[25]:


prediction_on_testing_data=regressor.predict(x_test)
prediction_on_testing_data


# In[26]:


prediction_on_testing_data.shape


# In[ ]:





# In[27]:


error_score=metrics.r2_score(Y_train,prediction_on_training_data)
print("error_score : ",error_score)


# In[28]:


error_score =metrics.r2_score(y_test,prediction_on_testing_data)
print("R squared error : ",error_score)


# In[29]:


#compare actual and predicted values
y_test=list(y_test)


# In[30]:


plt.plot(y_test,label='Actual Value')
plt.plot(prediction_on_testing_data,label='Predicted value',color='green')
plt.title('Actual price vs predicted price')
plt.xlabel('Number of values')
plt.ylabel('Gld price')
plt.legend()


# In[31]:


plt.plot(Y_train,label='Actual Value')
plt.plot(prediction_on_training_data,label='Predicted value',color='purple')
plt.title('Actual price vs predicted price')
plt.xlabel('Number of values')
plt.ylabel('Gld price')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




