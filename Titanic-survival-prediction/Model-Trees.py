#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import math


# In[63]:


train_data = pd.read_csv("Desktop/train.csv")
train_data.head()
#print(train_data)


# In[64]:



test_data = pd.read_csv("Desktop/test.csv")
test_data.head()


# In[65]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% women survived:", rate_women)


# In[66]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% men survived:", rate_men)


# In[67]:


from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Fare", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Desktop/my_submission2.csv', index=False)


# In[ ]:




