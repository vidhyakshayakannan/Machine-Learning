# Example 1-1. Training and running a linear model using Scikit-Learn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 


data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root+ "lifesat/lifesat.csv") # Loads csv data into data frame
X = lifesat[["GDP per capita (USD)"]].values # Only the values in the DataFrame will be returned, the axes labels will be removed.
y = lifesat[["Life satisfaction"]].values
print(X)

# Visualise data

lifesat.plot(kind='scatter', grid=True,
x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9]) # Set axis limits
plt.show()

# Select a linear model
model = LinearRegression()

# k nearest neighbours regression
model = KNeighborsRegressor(n_neighbors=20)

# Train the model
model.fit(X,y) # Finds the coefficients for the equation specified via the algorithm being used 

X_new = [[50683.32350972]]
print(model.predict(X_new))