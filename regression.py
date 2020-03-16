# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:57:53 2019

@author: lfpc9
"""

import pyodbc
import pandas as pd

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.0.6;DATABASE=LF;UID=LFADMIN;PWD=liases@123;autocommit=True')
cursor = cnxn.cursor()

sql_query = """ SELECT * FROM dbo.VW_LF_DATA vw_lf
INNER JOIN LF.DBO.MASTERDISTANCE dist
ON dist.projectid = vw_lf.PROJECT_ID
WHERE CITYID = 1
AND year(dos) = 2019 AND month(dos) in (1,2,3)
AND FLATTYPEID = 1
AND BOOKINGSTOP = 0
AND dist.clubid = 10
AND LOCATION = 'Andheri (E)'  """


p1 = pd.read_sql(sql_query, cnxn)

import googlemaps

from itertools import tee
import requests, json
import itertools

api = 'AIzaSyCV4QEDC3jXKfN6C3j4Ov0U4zN4nDzMNUE'

gmaps = googlemaps.Client(key = api)
##Andheri (E)
Station = (19.118869, 72.847941)
dist_s = []
for i in place:
    dist_s.append(gmaps.distance_matrix('Station', 'i') ['rows'] [0] ['elements'] [0])

Place = []
for i in p1.XY:
    Place.append(i)

place = tuple(Place)

p1['XY'] = p1[[float('PXVAL'), float('PYVAL')]].apply(lambda x: ','.join(x[x.notnull()]), axis=1)

##Star_Code
p1['XY'] = p1[['PXVAL','PYVAL']].apply(lambda x: ','.join(x.dropna().astype(float).astype(str)), axis=1)

p1.pop('XY')
##Correct Sequence

p1['XY'] = p1[['PXVAL','PYVAL']].apply(lambda x: ','.join(x.dropna().astype(float).astype(str)), axis=1)
Place = []
for i in p1.XY:
    Place.append(i)

place = tuple(Place)
Station = (19.280828, 72.856296)

##Alternate method

from math import sin, cos, sqrt, atan2, radians
R = 6373.0

dlist = []
xlat = radians(float(input ('lat:')))
ylong =radians(float(input ('long: ')))

list1 = []
for e in p1.PXVAL:
    list1.append(radians(e))
len(list1)

list2 = []
for f in p1.PYVAL:
    list2.append(radians(f))
len(list2)


for inde in range(len(list1)):
        dlon = ylong - list2[inde]
        dlat = xlat - list1[inde]
        a = (sin(dlat/2))**2 + cos(list1[inde]) * cos(xlat) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        dlist.append(distance)
        dlist

import numpy as np
p1['dist_s'] = dlist

p2 = p1.sort_values(by = 'dist_s')

p2['diff'] = (p2['DOS'] - p2['ENDDATE']).dt.days
  
##LINEAR REGRESSION

import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(p2['TOTALCOST'])

list(p2)

X = p2[['SALEABLE_FLAT_SIZE','STOREY', 'dist_s', 'distance', 'diff' ]]

X = p2[['SALEABLE_FLAT_SIZE','STOREY', 'dist_s', 'diff' ]]

y = p2[['TOTALCOST']]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test= train_test_split (X, y, test_size=0.3, random_state=198)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

lm.coef_

X.columns

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

sns.distplot(y_test-predictions)

##Assessing

from sklearn import metrics
from sklearn.metrics import r2_score


metrics.r2_score(y_test, predictions)

RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
mean_cost = p2.TOTALCOST.mean()
percent_error = RMSE/mean_cost
percent_error

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.regressionplots import abline_plot
import statsmodels.api as sm
import seaborn as sns
##With CBD

Index(['SALEABLE_FLAT_SIZE', 'STOREY', 'dist_s', 'distance', 'diff'], dtype='object')

array([[  1.59898987e+04,   2.13380287e+04,  -3.96847207e+05,
          1.11277172e+05,   1.33183699e+02]])

metrics.r2_score(y_test, predictions)
Out[37]: 0.72627973404977308

np.sqrt(metrics.mean_squared_error(y_test, predictions))

Out[38]: 1046284.0545064004

##Without CBD

(['SALEABLE_FLAT_SIZE', 'STOREY', 'dist_s', 'diff'], dtype='object')

lm.coef_

Out[5]: 
array([[  1.59952680e+04,   3.55377641e+04,  -3.28201274e+05,
          8.83287165e+01]])
metrics.r2_score(y_test, predictions)
Out[12]: 0.70078775627762591
   
np.sqrt(metrics.mean_squared_error(y_test, predictions))
Out[11]: 1093920.6243737121

   for i in p2['UNSOLD_UNIT_ASONDATE']:
    if i = 0:
        print(i)
    else:
        pass

p2.UNSOLD_UNIT_ASONDATE.value_counts()


##
y = p2[['TOTALCOST']]

X = p2[['SALEABLE_FLAT_SIZE','STOREY', 'dist_s', 'diff' ]]

X = p2[['dist_s']]
plt.scatter(X, y)

model = sm.OLS(y,X).fit()
predictions = model.predict(X)
model.summary()

sns.regplot(X, y)

corr = X.corr() 
sns.heatmap(corr)

corr.style.background_gradient(cmap = 'coolwarm')

np.corrcoef(X)

for i in p1.TOTALSUPPLY_SQFT_ASONDATE:
    if i == 1500:
        print(i)


p1 = p1.set_index('TOTALSUPPLY_SQFT_ASONDATE')
p1 = p1.drop(<1500, axis = 0)


stat, p = shapiro(residuals)
print('Statistics = %.3f, p = %.3f' % (stat, p))

stat, p = normaltest(residuals)
print('Statistics = %.3f, p = %.3f' % (stat, p))

stat, p = shapiro(residuals)
print('Statistics = %.3f, p = %.3f' % (stat, p))










