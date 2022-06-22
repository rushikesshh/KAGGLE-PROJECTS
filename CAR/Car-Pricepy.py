import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\RUSHIKESH\\Downloads\\archive (10)\\CarPrice_Assignment.csv")

data.shape

data.info()

data.fueltype.value_counts()

data.drop(columns=['car_ID','CarName',], axis = 1, inplace=True)




# CONVERTING THE CATEGORICAL VARIABLE INTO NUMERICAL VARIABLE :--

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['fueltype'] = label_encoder.fit_transform(data.fueltype)
data['aspiration'] = label_encoder.fit_transform(data.aspiration)
data['doornumber'] = label_encoder.fit_transform(data.doornumber)
data['carbody'] = label_encoder.fit_transform(data.carbody)
data['drivewheel'] = label_encoder.fit_transform(data.drivewheel)
data['enginelocation'] = label_encoder.fit_transform(data.enginelocation)
data['enginetype'] = label_encoder.fit_transform(data.enginetype)
data['cylindernumber'] = label_encoder.fit_transform(data.cylindernumber)
data['fuelsystem'] = label_encoder.fit_transform(data.fuelsystem)


data.isnull().sum()

correlation = data.corr()

# WE WILL TAKE THOSE VARIABLES WHICH HAVING THE GOOD CORRELATION WITH OUTPUT VARIABLE :--
 

data.drop(columns=["symboling","aspiration","doornumber",
                   "carbody", "carheight", "peakrpm" ,"compressionratio" ],
                    axis = 1 , inplace=True)

data
        
# VISUALIZATION :--


# SCATTER-PLOTS :--

fig, ax = plt.subplots(2,2 , figsize=[12,8])

ax[0][0].scatter(data.price , data.horsepower)
ax[0][0].set_xlabel("Price")
ax[0][0].set_xlabel("HorsePower")


ax[0][1].scatter(data.price , data.enginesize)
ax[0][1].set_xlabel("Price")
ax[0][1].set_xlabel("Engine_size")

ax[1][0].scatter(data.price , data.curbweight)
ax[1][0].set_xlabel("Price")
ax[1][0].set_xlabel("Curbe_weight")

ax[1][1].scatter(data.price , data.highwaympg)
ax[1][1].set_xlabel("Price")
ax[1][1].set_xlabel("mpg_highway")

# plt.scatter(data.price , data."variable")



# HIST-PLOTS :--

fig, ax = plt.subplots(2,2 , figsize=[12,8])

ax[0][0].hist(data.price , edgecolor='black')
ax[0][0].set_title("Price")

ax[0][1].hist(data.carlength , edgecolor='black')
ax[0][1].set_title("Car_length")

ax[1][0].hist(data.horsepower , edgecolor='black')
ax[1][0].set_title("Horsepower")

ax[1][1].hist(data.citympg , edgecolor='black')
ax[1][1].set_title("CityMPG")
    
# plt.hist(data."Variable", edge_color='black)


# BOX-PLOTS :--

fig, ax = plt.subplots(2,1 , figsize=[10,8])
    
sn.boxplot(data.price , ax=ax[0])
ax[0].set_title("Boxplot - Price")

sn.boxplot(data.horsepower, ax=ax[1])
ax[1].set_title("Boxplot - Horsepower")
    


fig, ax = plt.subplots(2,1 , figsize=[10,8])
    
sn.boxplot(data.carlength , ax=ax[0])
ax[0].set_title("Boxplot - Car-Length")

sn.boxplot(data.curbweight, ax=ax[1])
ax[1].set_title("Boxplot - curbweight")
    

# sn.boxplot(data,"variable")
    
    
    
# DIVIDING THE DATA INTO TRAIN AND TEST :--

from sklearn.model_selection import train_test_split

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)
    

# MODEL-BUILDING :--

# AS WE HAVE LESS AMOUNT OF DATA WE WILL GO WITH K-FOLD CROSS VALIDATION :--

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict


# 1. LINEAR-REGRESSION :-

from sklearn.linear_model import LinearRegression


model_1 = LinearRegression()
num_folds = 10   
kfold = KFold(n_splits=num_folds) 

result_1 = cross_val_score(model_1,x_train, y_train ,cv=kfold)

result_1.mean()*100

model_1.fit(x_train, y_train)

pred_1 = model_1.predict(x_test) 

def rmse(actual , pred):
    rmse = np.sqrt(np.mean(actual-pred)**2)
    return rmse
    

rmse_model_1 = rmse(y_test, pred_1)   
rmse_model_1
 

# 2. DECISION TREE REGRESSOR :-
    
from sklearn.tree import DecisionTreeRegressor

model_2 = DecisionTreeRegressor()
result_2 = cross_val_score(model_2, x_train, y_train, cv=kfold)  
result_2.mean()*100

model_2.fit(x_train, y_train) 
pred_2 = model_2.predict(x_test)   

rmse_model_2 = rmse(y_test, pred_2)
rmse_model_2    


# 3. K-Nearest Neighbor (KNN) :-

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# for using KnN we have to normalize our data :-
# here we only normalize the input data :-

scalar = StandardScaler()

X_norm = scalar.fit_transform(data.iloc[:,:-1])


x_train_norm, x_test_norm, y_train, y_test = train_test_split(X_norm, Y, test_size=0.25)

k_values = np.array(range(3,30,2))
param_grid = dict(n_neighbors=k_values)
param_grid

model = KNeighborsRegressor()

model_3 = GridSearchCV(estimator = model, param_grid = param_grid)
model_3.fit(x_train_norm, y_train)

model_3.best_score_    
model_3.best_params_

pred_3 = model_3.predict(x_test_norm)

rmse_model_3 = rmse(y_test, pred_3)    
rmse_model_3    
    
    
    
# 4. Support Vector Machine (SVM) :-

from sklearn.svm import SVR

model = SVR()

kernals = ['rbf', 'linear','Gaussian']
param_grid = dict(kernel=np.array(kernals))
param_grid
    
    
model_4 = GridSearchCV(estimator = model, param_grid = param_grid)
model_4.fit(x_train, y_train)    

model_4.best_params_    
model_4.best_score_

pred_4 = model_4.predict(x_test)    

rmse_model_4 = rmse(y_test, pred_4)
rmse_model_4    
    
    
# 5. Ridge Regularization (L2) :-

from sklearn.linear_model import Ridge

model = Ridge()
alpha = np.array(range(1,20))
param_grid = dict(alpha = alpha)
param_grid   
    
model_5 = GridSearchCV(estimator=model, param_grid=param_grid)
model_5.fit(x_train , y_train)

model_5.best_params_    
model_5.best_score_

pred_5 = model_5.predict(x_test)    

rmse_model_5 = rmse(y_test, pred_5)
rmse_model_5    



# 6. Lasso Regularization (L1) :- 

from sklearn.linear_model import Lasso   

model_6 = Lasso()    
model_6.fit(x_train, y_train)

pred_6=model_6.predict(x_test)    

rmse_model_6 = rmse(y_test, pred_6)
rmse_model_6




# 7. Bagging Regressor :-

from sklearn.ensemble import BaggingRegressor    
from sklearn.tree import DecisionTreeRegressor    

model = DecisionTreeRegressor()
num_trees = np.array([100,150,200,250,300,350,400,500])

param_grid = dict(n_estimators = num_trees, base_estimator=[model])
param_grid

model_7 = GridSearchCV(estimator= BaggingRegressor(), param_grid=param_grid)

model_7.fit(x_train, y_train)    

model_7.best_params_
model_7.best_score_
pred_7 = model_7.predict(x_test)    

rmse_model_7=rmse(y_test, pred_7)    
rmse_model_7    
    

    
# 8. Random Forest Tecnique :-

from sklearn.ensemble import RandomForestRegressor    

model = RandomForestRegressor() 
num_trees = np.array([100,150,200,250,300,350,400,500])
max_features = np.array([3,4,5,6])
                         
param_grid = dict(n_estimators = num_trees , max_features = max_features )
param_grid

model_8 = GridSearchCV(estimator=model, param_grid=param_grid)
model_8.fit(x_train, y_train) 

model_8.best_params_ 
model_8.best_score_

pred_8 = model_8.predict(x_test)    

rmse_model_8 = rmse(y_test, pred_8)    
rmse_model_8    
    
    
 
# 9. AdaBoost Technique :-

from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor()    

n_estimators = np.array(range(100,500,50))
learning_rate = np.array([0.1,0.5,1,0.7])  

param_grid = dict(n_estimators = n_estimators, learning_rate = learning_rate)
param_grid  

model_9 = GridSearchCV(estimator=model, param_grid=param_grid)    
model_9.fit(x_train, y_train)    

model_9.best_params_
model_9.best_score_    

pred_9 = model_9.predict(x_test)    

rmse_model_9 = rmse(y_test, pred_9)
rmse_model_9    
    
    

# 10. Extreme Gradient Boosting :-

from xgboost import XGBRegressor

model = XGBRegressor() 
n_estimators = np.array(range(100, 500,50))
max_depth = np.array([3,5,7,6])  
learning_rate = np.array([0.1,0.5,1,0.7])

param_grid = dict(n_estimators = n_estimators, max_depth=max_depth, learning_rate=learning_rate)
param_grid 

model_10 = GridSearchCV(estimator=model, param_grid=param_grid)
model_10.fit(x_train, y_train)    

model_10.best_params_
model_10.best_score_    

pred_10 = model_10.predict(x_test)    

rmse_model_10 = rmse(y_test , pred_10)
rmse_model_10    
    

    

rmse_ = { 'MODEL' : ('LINEAR-REGRESSION' , 'DECISION TREE REGRESSOR', 
                             'K-Nearest Neighbor (KNN)' ,'Support Vector Machine (SVM)'
                             ,'Ridge Regularization (L2)' , 'Lasso Regularization (L1)' ,
                             'Bagging Regressor' ,'Random Forest Tecnique' ,
                             'AdaBoost Technique', 'Extreme Gradient Boosting'),
         
         'RMSE_VALUE' : (rmse_model_1, rmse_model_2, rmse_model_3, rmse_model_4,
                            rmse_model_5, rmse_model_6, rmse_model_7, rmse_model_8,
                            rmse_model_9, rmse_model_10)}
    
    
RMSE_DF = pd.DataFrame(rmse_)
    
RMSE_DF.sort_values(RMSE_DF.RMSE_VALUE, ascending=True, inplace=True)    
    
    
    
    
# WE ARE GETTING BEST RMSE VALUE FOR LINEAR REGRESSION SO WE WILL BUILD OUR FINAL MODEL ON THAT :-

final_model = LinearRegression()

final_model.fit(X,Y)    


    


