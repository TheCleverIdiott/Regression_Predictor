#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Loading the dataset to pandas DataFrame from remote link
data = "https://raw.githubusercontent.com/TheCleverIdiott/Score_Predictor/main/Dataset"   
students_data = pd.read_csv(data)
students_data.head()

students_data.shape
students_data.info()

#Checking for any missing values in the dataset
students_data.isnull().sum()

# Plotting the distribution of scores
students_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# X contains independent variable and y contains dependent variable
X = students_data.iloc[:, :-1].values
y = students_data.iloc[:, 1].values

print(y)

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#TRAINING THE MODEL
from  sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)

#For retrieving the slope (coefficient of x)
print(model.coef_)

#For retrieving the intercept
print(model.intercept_)

# Plotting the regression line
line = model.coef_*X+model.intercept_
# Plotting for the test data
plt.scatter(X, y,label="True values")
plt.plot(X, line,color="red",label="Predicted values")
plt.title('Simple Linear Regression')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.legend()
plt.show()

#Making Predictions
y_pred = model.predict(X_test)
y_pred

#Making prediction for 9.25 hours/day
hours=9.25
input_data_as_numpy_array=np.asarray(hours)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
own_prediction=model.predict(input_data_reshaped)
print("No. of Hours studied:",hours)
print("Predicted Score:",own_prediction)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Conculsion: You can see that the value of root mean squared error is 4.64, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48. This means that our algorithm did a decent job


