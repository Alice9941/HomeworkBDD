#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression #remember, this is linear regression model from sk-learn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


df = pd.read_csv('survey.csv')
df.head()


# In[3]:


df2 = pd.read_csv('steps.csv' , sep=";")
df2.head()


# In[4]:


df3 = df2.merge(df, on="id")


# In[ ]:





# In[5]:


df3.head()


# In[6]:


df3.dropna


# In[7]:


# Geen idee wat ik moet doen en hoe ik het moet doen. 


# In[8]:


sns.distplot(df["weight"], kde=False) 


# In[9]:


sns.distplot(df["height"], kde=False) 


# In[10]:


##df_subset = df[['area', 'rooms', 'price']]
##sns.pairplot(df_subset)
##plt.show()


# In[ ]:





# In[ ]:


#Om de relatie te gaan zien 


# In[11]:


#residual = house['price_p'] - house['price'] #Subtracting Y'-Y (Y' = predicted) gets us the residual
#sns.scatterplot(x='price',y='price_p',data=df)
#plt.xlim(0, 1400000) #This sets the x-axis limits to (0, 5e10 = 140000000000)
#plt.ylim(0, 1400000) #Ditto for y-axis. I want both axes to have the same length, so we can compare them
#plt.plot([0, 1.5e6], [0, 1.5e6], color='red', lw=3) #This draws the straight red line, you can leave this out if you wish
#plt.xlabel('Price')
#plt.ylabel('Price (predicted)')
#plt.show()


# In[ ]:





# In[ ]:


#Variablen aanpassen natuurlijk, maar dit is wat we nodig hebben voor de scatterplot


# In[12]:


##corr = df[['price', 'area', 'rooms']].corr()
##corr


# In[ ]:


#Om de correlatie nummers te vinden 


# In[ ]:


from sklearn.linear_model import LinearRegression #we need this specific model from sk-learn

x = df[['area']] #to use sk-learn, we need to create lists of the two variables
y = df['price']

lm = LinearRegression() #this creates a new LR model
lm.fit(x, y) #this "fits" the model

b0 = lm.intercept_ #lm.coef_ gets a list of coefficients (excluding the intercept). [0] gets the actual number from the list
b1 = lm.coef_[0] #gets the intercept

print(f"The regression line is equal to y = {b0:.2f} + {b1:.2f}X") #.0f formats to 2 decimals.


# In[ ]:


# OM de linear regression te vinden. Again variabelen aanpassen

