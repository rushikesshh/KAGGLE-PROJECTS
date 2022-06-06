import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from xgboost import XGBRegressor
import streamlit as st

st.title('Medical Claim Calculater ')

st.sidebar.title("User input parameter")
  
def user_input() :
    
    age = st.sidebar.number_input("ENTER YOUR AGE")                                                            
    sex = st.sidebar.selectbox(  'FOR_SEX_________________________________ ||  FEMALE = 0 , MALE = 1 || ',[0,1])
    bmi =    st.sidebar.number_input("enter bmi ")
    children = st.sidebar.selectbox("NUMBER OF CHILDREN ",[0,1,2,3,4,5,6])
    smoker = st.sidebar.selectbox(  'SMOKER____________________________________ ||  NO = 0 , YES = 1 || ',[0,1])
    region = st.sidebar.selectbox("ENTER YOUR REGION _____________________ || northeast = 0 , northwest = 1 , southeast = 2 , southwest = 3 ",                                                                                                                  [0,1,2,3])
    
    
    data = { 'age' : age,
             'sex' :sex,
             ' bmi' : bmi,
             'children' : children,
             'smoker' : smoker,
             'region' : region 
           }
    
    features = pd.DataFrame(data,  index=[0])
    
    return features


df = user_input()
st.subheader("Data Of User :-")
st.write(df)


data = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\expenses.csv")

clean_data = data.drop_duplicates()

clean_data.reset_index(drop=True)



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

clean_data['sex'] = label_encoder.fit_transform(clean_data['sex'])
clean_data['smoker'] = label_encoder.fit_transform(clean_data['smoker'])
clean_data['region'] = label_encoder.fit_transform(clean_data['region'])

X = clean_data.iloc[:,:-1]
Y = clean_data.iloc[:,-1]


final_model = model = XGBRegressor(learning_rate=0.1, max_depth = 3, n_estimators = 200)

final_model.fit(X,Y)

prediction = final_model.predict(df)

st.subheader("Charges :-")
st.write(prediction)







































                                  
                                              