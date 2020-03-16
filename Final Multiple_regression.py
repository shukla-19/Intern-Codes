# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:57:53 2019

@author: lfpc9
"""

import pyodbc
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
 

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.regressionplots import abline_plot
import statsmodels.api as sm
import seaborn as sns

location = ['Dombivali (E)',  
'Panvel',             
'Thane (W)',          
'Badlapur (E)',     
'Taloja',             
'Mira Road (E)',      
'Andheri (E)',        
'Malad (W)',          
'Kandivali (W)',     
'Prabhadevi',          
'Mahim (W)']

location = ['Dombivali (E)',  
'Panvel']


Station = [(19.217318, 73.086261), (18.991121,73.120769) ,(19.186545, 72.975680),
           (19.166682,73.238696), (19.079845, 73.088287), (19.280828, 72.856296),
           (19.118869, 72.847941), (19.186910, 72.848112), (19.203127, 72.851389),
           (19.007634, 72.835989), (19.946094, 72.823592)]

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
AND LOCATION = 'Panvel'  """


p1 = pd.read_sql(sql_query, cnxn)
Station = (19.186545, 72.975680)

##Alternate method


R = 6373.0

dlist = []
xlat = radians(Station[0])
ylong = radians(Station[1])

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


p1['dist_s'] = dlist

p2 = p1.sort_values(by = 'dist_s')

p2['diff'] = (p2['DOS'] - p2['ENDDATE']).dt.days
  
##LINEAR REGRESSION

sns.distplot(p2['TOTALCOST'])

X = p2[['SALEABLE_FLAT_SIZE','STOREY', 'diff', 'dist_s']]

y = p2[['TOTALCOST']]

X_train, X_test, y_train, y_test= train_test_split (X, y, test_size=0.3, random_state=71)

lm = LinearRegression()

model = lm.fit(X_train, y_train)

model = lm.fit(X, y)

print(lm.intercept_)

lm.coef_

X.columns

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

sns.distplot(y_test-predictions)

model.score(X_test, y_test)

kf = KFold(n_splits=10)

kf.get_n_splits(X)

cross_val_score(model, df, y, cv=5)

rfe_1 = RFE(lm, 1)
fit = rfe_1.fit(X, y)
fit.n_features_
fit.ranking_
fit.score


##Assessing
metrics.r2_score(y_test, predictions)

RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
mean_cost = p2.TOTALCOST.mean()
percent_error = RMSE/mean_cost
percent_error

model = sm.OLS(y,X).fit()
prediction = model.predict(X)
model.summary()



corr = X.corr() 
corr
sns.heatmap(corr)
np.corrcoef(X)

#Results Value -
##Andheri(E) (19.118869, 72.847941)
 metrics.r2_score(y_test, predictions)
Out[7]: 0.91063574964130667
percent_error
Out[8]: 0.137152711213231

##Dombivali(E) 
metrics.r2_score(y_test, predictions)
Out[45]: 0.8996576419894573
percent_error
Out[49]: 0.14734407494126883
   
## Panvel

metrics.r2_score(y_test, predictions)
Out[64]: 0.88103346121154402
percent_error
Out[69]: 0.27687009784043504

###Thane(W)

metrics.r2_score(y_test, predictions)
Out[74]: 0.91195015031890447RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
percent_error
Out[75]: 0.26868139635313437

##Badlapur (E)
metrics.r2_score(y_test, predictions)
Out[26]: 0.88824514344524852
percent_error
Out[27]: 0.11463643032870884

##Taloja
metrics.r2_score(y_test, predictions)
Out[33]: 0.94333967490968917
percent_error
Out[31]: 0.11845862916452402

##Mira road
metrics.r2_score(y_test, predictions)
Out[39]: 0.95587843714846754
percent_error
Out[40]: 0.071750041972812345
#Filtered
metrics.r2_score(y_test, predictions)
Out[30]: 0.92918383927441106
percent_error
Out[31]: 0.083129179726988023
#

##Malad(w)
metrics.r2_score(y_test, predictions)
Out[46]: 0.93534576115395507
percent_error
Out[47]: 0.12356462629928955

##Kandivali(W)
metrics.r2_score(y_test, predictions)
Out[52]: 0.92224354395263242
percent_error
Out[51]: 0.16289421921483213

##Marine Lines
metrics.r2_score(y_test, predictions)
Out[56]: 0.98739600434690122
percent_error
Out[57]: 0.20025392893214553

##Prabhadevi
 metrics.r2_score(y_test, predictions)
Out[62]: 0.99627585773951444
percent_error
Out[63]: 0.13564686291703523
