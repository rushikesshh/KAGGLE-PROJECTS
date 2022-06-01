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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# In[ ]:


st.title("SPACESHIP")
st.title("|| Allowed for onboard or not  ||")

st.sidebar.title("User input parameter")

def user_input ():
 
    HomePlanet = st.sidebar.selectbox('FOR_HOME_PLANET__________________________ ||    EARTH = 0 , EUROPA = 1 , MARS = 2    || ',[0,1,2])
    CryoSleep = st.sidebar.selectbox(  'FOR_CRYOSLEEP____________________________ ||  FALSE = 0 , TRUE = 1 || ',[0,1])
    Destination = st.sidebar.selectbox('FOR_DESTINATION___________________________   55_Cancri_e  =  0  || PSO_J318.5-22 = 1 ||  TRAPPIST-1e                                                                                       = 2',[0,1,2])
                                                                                      
    Age = st.sidebar.number_input("ENTER YOUR AGE ___________________________")
    VIP = st.sidebar.selectbox(' FOR_VIP___________________________________ ||  FALSE = 0 , TRUE = 1 || ',[0,1])
    
    RoomService = st.sidebar.number_input("ENTER_ROOM_SERVICE _____________________")
    FoodCourt   =  st.sidebar.number_input("ENTER_FOOD_COURT _______________________")
    ShoppingMall = st.sidebar.number_input("ENTER_SHOPPING_MALL ____________________")
    Spa          = st.sidebar.number_input("ENTER_SPA ______________________________")
    VRDeck       = st.sidebar.number_input("ENTER_VRDECK ___________________________")
    
                                       
    
    data = { "HomePlanet" : HomePlanet,
             "CryoSleep" :  CryoSleep,
             "Destination" : Destination,
             "Age" : Age,
             "VIP" : VIP,
             "RoomService" : RoomService,
             "FoodCourt" : FoodCourt,
             "ShoppingMall" : ShoppingMall ,
             "Spa" : Spa,
             "VRDeck" : VRDeck
           }
    
    features = pd.DataFrame(data, index=[0])
    
    return features

df = user_input()
st.subheader("ENTERED DATA OF USER")

st.write(df)

data = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\spaceship-titanic\\train.csv")

label_encoder = LabelEncoder()

data['VIP'] = label_encoder.fit_transform(data.VIP) 
data['HomePlanet'] = label_encoder.fit_transform(data.HomePlanet) 
data['CryoSleep'] = label_encoder.fit_transform(data.CryoSleep) 
data['Transported'] = label_encoder.fit_transform(data.Transported) 
data['Destination'] = label_encoder.fit_transform(data.Destination) 

data.drop(["PassengerId","Cabin","Name"], axis=1, inplace=True)

data_1 = data.dropna()
data_1.reset_index(drop=True, inplace=True)

X = data_1.iloc[:,:-1]
Y = data_1.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)


Final_model = XGBClassifier(n_estimators=100, max_depth=4)
Final_model.fit(X, Y)

prediction = Final_model.predict(df)


st.subheader("Result")
st.write(prediction)

if prediction == 1 :
    print (st.success("You will Onboard on spaceship"))
else:
    print (st.error("You will not allow to Onboard on spaceship"))

    


    
    