#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sn
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[ ]:


st.title("Deployment")
st.title("|| Amount Spent Yearly on E-Commerce Website ||")

st.sidebar.title("User input parameter")

def user_input ():
    session_len = st.sidebar.number_input("The session len average ")
    time_spend_on_app = st.sidebar.number_input("The time spend on application")
    time_spend_on_website = st.sidebar.number_input("The time spend on website")
    len_of_membership = st.sidebar.number_input("The length of membership ")
    
    data = { "Avg_session_len" : session_len,
             "Time_app" :  time_spend_on_app,
             "Time_website" : time_spend_on_website,
             "Len_membership" : len_of_membership }
    
    features = pd.DataFrame(data, index=[0])
    
    return features

df = user_input()
st.subheader("User Input Parameter")

st.write(df)

data = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\Ecommerce Customers (2)")

data.drop(["Email", "Address","Avatar"], inplace=True, axis=1)

data_2 = data.rename({"Avg. Session Length": "Avg_session_len",
                   "Time on App": "Time_app",
                   "Time on Website" : "Time_website",
                   "Length of Membership" : "len_membership",
                   "Yearly Amount Spent" : "Yearly_spent"}, axis=1
                  )

X = data_2.iloc[:,:-1]
Y = data_2.iloc[:,-1]

alpha = np.array(range(1,30))
param_grid = dict(alpha = alpha)


final_model =  Lasso()

grid = GridSearchCV(estimator = final_model, param_grid = param_grid)
grid.fit(X,Y)



prediction = grid.predict(df)


st.subheader("Amount_spend_yearly")
st.write(prediction)


































    
    
    