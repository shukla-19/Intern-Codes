import pyodbc
import pandas as pd





cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.0.6;DATABASE=LF;UID=LFADMIN;PWD=liases@123;autocommit=True')
cursor = cnxn.cursor()

sql_query = """SELECT * FROM dbo.VW_LF_DATA
            WHERE CITYID = 1
            AND year(dos) = 2019 AND month(dos) in (1,2,3)
            AND FLATTYPEID = 1
            AND BOOKINGSTOP = 0"""


input_data = pd.read_sql(sql_query, cnxn)

#Pandas Revision on Rows and Columns
input_data.shape
input_data.head(3)
input_data['QTR_ID'].describe()

input_data.iloc[1200:1220, 221:224]
input_data.iloc[1:31, 93:95]

##Filtering by Andheri(e) and Flat ID

p1 = input_data.loc[(input_data['LOCATION'] == 'Andheri (E)') & 
                      (input_data['FLAT_ID'] == 3)]

p1
len(p1)

##Distance from station

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
p1['distance'] = dlist

  p2 = p1.sort_values(by = 'dist_station')
  
s = float(input('size:'))
for i in p2.SALEABLE_FLAT_SIZE:
    print(abs(i - s))
    

p2['size'] = [(abs(i - s)) for i in p2.SALEABLE_FLAT_SIZE]

##LINEAR REGRESSION

import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(p2['TOTALCOST'])

list(p2)

X = p2[['SALEABLE_FLAT_SIZE','STOREY', 'DISTRANGEID', 'distance', 'diff' ]]

y = p2[['TOTALCOST']]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test= train_test_split (X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

lm.coef_

X.columns

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

sns.distplot(y_test-predictions)

p2['diff'] = (p2['ENDDATE'] - p2['DOS']).dt.days


for i in columns.DISTANCERANGE:
    if i == '16 - 18 KM':
        
import googlemaps

from itertools import tee
import requests, json

api = 'AIzaSyCV4QEDC3jXKfN6C3j4Ov0U4zN4nDzMNUE'

gmaps = googlemaps.Client(key = api)



dist_s = gmaps.distance_matrix('Station', 'Place', mode='transit' ['rows'] [0]
['elements'] [0], ['distance'] ['value'])

















