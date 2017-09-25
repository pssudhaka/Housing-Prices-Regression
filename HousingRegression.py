import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# load the values for the two datasets
train= pd.read_csv('C:\\untitled3\\train.csv')
test = pd.read_csv('C:\\untitled3\\test.csv')

#view the data and look for missing values

train.head()
test.head()

# view the panda dataframe
train.info()

# drop all the data with two few datapoints

# check the data again

train.head()

# view the shape of the data

print(train.shape)
print(test.shape)

# generate some plots of the data

plt.hist(train.SalePrice,color='red')
plt.show()

print("Sale price skew is ",train.SalePrice.skew())

# boxplot for neighbourhoods
plt.figure(figsize=(13,7))
sns.boxplot(x='Neighborhood',y='SalePrice',data=train)
xt=plt.xticks(rotation=45)

#boxplot for SaleTypes and conditions
plt.figure(figsize=(13,7))
sns.boxplot(x='SaleType',y='SalePrice',data=train)
sns.boxplot(x='SaleCondition',y='SalePrice',data=train)

number=train.select_dtypes(include=[np.number])
number.dtypes

# now lets check a correlation
corr=number.corr()
corr

# check for more data outliers so that nothing has been missed

plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
plt.title("Outliers")
plt.xlabel("Area")
plt.ylabel("Sale Price")

plt.show()

#Now lets give some basic stats about the SalePrice
print("Stats about SalePrice")
print(train["SalePrice"].describe());

# now fill holes in the dataframes
test.fillna(method='bfill',inplace=True)
train.fillna(method='bfill',inplace=True)

# redo the scatter plot to check that the holes have filled
plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
plt.title("Outliers")
plt.xlabel("Area")
plt.ylabel("Sale Price")

plt.show()

# Now lets start training the regression models
from sklearn.model_selection import train_test_split
x_axis=train.drop('SalePrice',axis=1)
y_axis=train.SalePrice

x_axis_train,x_axis_test,y_axis_train,y_axis_test=train_test_split(x_axis,y_axis)

# first model: a simple linear regression
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression(normalize=False)
linear_reg.fit(x_axis_train,y_axis_train)
predictions = linear_reg.predict(x_axis_test)

# second model try a simple logistic regression
from sklearn.linear_model import logistic
logistic_reg=logistic(normalize=False)
logistic_reg.fit(x_axis_train,y_axis_train)
logistic_predictions=logistic_reg.predict(x_axis_test)
