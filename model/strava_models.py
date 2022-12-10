import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score,median_absolute_error
def plot(prediction):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,7)) 
    sns.distplot(y_test.values,label='test values', ax=ax1)
    sns.distplot(prediction ,label='prediction', ax=ax1)
    ax1.set_xlabel('Distribution plot')
    
    ax2.scatter(y_test,prediction, c='m',label='predictions')
    ax2.plot(y_test,y_test, c='black',label='Observed value')
   
    ax2.set_xlabel('test value')
    ax2.set_ylabel('estimated $\log(radius)$')
    ax1.legend()
    ax2.legend()
    ax2.axis('scaled') #same x y scale
    plt.show()


activities = pd.read_csv('/Users/ajit/Desktop/ELT-strava/activities.csv')


dropNumColumn = ['name','type','start_date_local','start_time','average_speed','max_speed','max_speed']
activities = activities.drop(dropNumColumn, axis = 1)
activities['average_heartrate'].fillna(0, inplace=True)
# print(activities.info())
# print(activities.describe())

# define our input and output variable 
X = activities.drop(['distance'], axis = 1)
y = activities[['distance']]


# split the dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1) #from sklearn library

# create an instance of our model
MR_model = LinearRegression()

# fit the model
MR_model.fit(X_train, y_train)

#coefficient of the model and the intercept
intercept = MR_model.intercept_[0] 
coefficient = MR_model.coef_[0] #nested list

print("The intercept of our model is {:.4}".format(intercept))
print('-'*100)

for cf in zip(X.columns,coefficient):
    print('The coefficient for {} is {:.4}'. format(cf[0],cf[1]))

# ----------------------------
# define the input
X2 = sm.add_constant(X) #fromstatsmodel.api

# create a Ordinary least sqaure model
model = sm.OLS(y, X2)

#fit the data
estimate = model.fit()

# runnning pagan's test
_, pval, _, f_pval = diag.het_breuschpagan(estimate.resid, estimate.model.exog)
print('p value =', pval,'; fp value =', f_pval)
print('-'*100)
# print the results of the test
if pval > 0.05:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
else:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


# ----------
import pylab

# check for the normality of the residuals
sm.qqplot(estimate.resid, line='s')
pylab.show()

# also check that the mean of the residuals is approx. 0.
mean_residuals = sum(estimate.resid)/ len(estimate.resid)
print("The mean of the residuals is {:.4}".format(mean_residuals))

#-------
import math
Y_predict = MR_model.predict(X_test)
# calculate the mean squared error
model_mse = mean_squared_error(y_test, Y_predict)

# calculate the mean absolute error
model_mae = mean_absolute_error(y_test, Y_predict)

# calulcate the root mean squared error
model_rmse =  math.sqrt(model_mse)

# calculate the explained variance score
model_evs = explained_variance_score(y_test, Y_predict)

model_meae = median_absolute_error(y_test, Y_predict)

# display the output
print("MSE {:.5}".format(model_mse))
print("MAE {:.5}".format(model_mae))
print("RMSE {:.5}".format(model_rmse))
print("EVS {:.5}".format(model_evs))
print("MEAE {:.5}".format(model_meae))

# ----
model_r2 = r2_score(y_test, Y_predict)
print("R2: {:.5}".format(model_r2))

plot(Y_predict)