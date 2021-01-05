# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ____
# ## Logistic Regression Advertising report
# ### Author: Alex Fields
# 
# In this project I will be showcasing my LogisticRegression skills by using an advertising csv file from Kaggle. 
# This objective is to see if we can accurately predict if the user will click on the add based on the below features. 
# 
# #### Features from the Dataset: 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
ad_data = pd.read_csv('advertising.csv')
ad_data.head()


# %%
ad_data.describe()


# %%
ad_data.info()


# %%
sns.histplot(data=ad_data,x='Age', bins=30)


# %%
sns.jointplot(x="Age", y='Daily Time Spent on Site' , data=ad_data, kind='kde', color='red', shade=True, cmap='Reds')


# %%
sns.jointplot(data=ad_data, x="Daily Time Spent on Site", y='Daily Internet Usage')


# %%
#may take time to run depending on size of data
sns.pairplot(data=ad_data, hue='Clicked on Ad')


# %%
from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)


# %%
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# %%
logmodel.fit(X_train, y_train)


# %%
predictions = logmodel.predict(X_test)


# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


