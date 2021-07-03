## import package
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load data
df = pd.read_csv("regrex1.csv")

## plot scatter
plt.scatter(x=df.x, y=df.y)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("py_orig.png")

## model the data
linear_regressor = LinearRegression()
X = df.iloc[:, 1].values.reshape(-1, 1)
Y = df.iloc[:, 0].values.reshape(-1, 1)
fit = linear_regressor.fit(X, Y)

## predict y
Y_pred = linear_regressor.predict(X)

## plot scatter with predicting value
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("py_lm.png")