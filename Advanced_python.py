#!/usr/bin/env python
# coding: utf-8

## convert jupyter notebook to sript
# jupyter nbconvert --to script notebookname.ipynb


## import package
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy import stats

import sys

## load data
df = pd.read_csv(sys.argv[1])
print("loading {}".format(sys.argv[1]))


## take out the file name
basename = os.path.splitext(sys.argv[1])


## plot scatter
plt.scatter(x=df.x, y=df.y)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Advanced_{}_scatter.png".format(basename[0]))
print("saving Advanced_{}_scatter.png".format(basename[0]))


## model the data
fit = stats.linregress(df.x, df.y)


## predict y
Y_pred = fit[0]*df.x + fit[1]


## plot scatter with predicting value
plt.scatter(x=df.x, y=df.y)
plt.plot(df.x, Y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Advanced_{}_scatter_lm.png".format(basename[0]))
print("saving Advanced_{}_scatter_lm.png".format(basename[0]))


## finish
print("done")