# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:53:56 2019

@author: lfpc9
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:57:53 2019

@author: lfpc9
"""

import pyodbc
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.regressionplots import abline_plot
import statsmodels.api as sm
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

from math import sin, cos, sqrt, atan2, radians

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.0.6;DATABASE=LF;UID=LFADMIN;PWD=liases@123;autocommit=True')
cursor = cnxn.cursor()


def calc_radial_dist(source_lat, source_long, dest_lat, dest_long):
    R = 6373.0
    
    dlist = []
    xlat = radians(float(source_lat))
    ylong =radians(float(source_long))
    
    dlon = ylong - radians(float(dest_long))
    dlat = xlat - radians(float(dest_lat))
    a = (sin(dlat/2))**2 + cos(radians(float(dest_lat))) * cos(xlat) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance
  


sql_query = """ SELECT * FROM dbo.VW_LF_DATA vw_lf
            INNER JOIN LF.DBO.MASTERDISTANCE dist
            ON dist.projectid = vw_lf.PROJECT_ID
            WHERE CITYID = 1
            AND year(dos) = 2019 AND month(dos) in (1,2,3)
            AND FLATTYPEID = 1
            AND BOOKINGSTOP = 0
            AND dist.clubid = 10 """

input_data = pd.read_sql(sql_query, cnxn)

sql_query = """ SELECT * FROM LF.DBO.MST_TRANSIT_DISTANCE """
dist_s = pd.read_sql(sql_query, cnxn)

dist_s_pivot = pd.pivot_table(dist_s, index = ['LF_PROJECT_ID'], values = ['DISTANCE_KM', 'TIME_TO_REACH_MIN'], aggfunc = np.min)
dist_s_pivot = pd.DataFrame(dist_s_pivot.to_records())

input_data=pd.merge(input_data, dist_s_pivot, how='inner', left_on= 'PROJECT_ID', right_on = 'LF_PROJECT_ID')


sql_query = """ SELECT * FROM LF.DBO.MST_TRANSIT_STATION WHERE IS_ACTIVE = 1 """
station_list = pd.read_sql(sql_query, cnxn)

min_dist_array = []
for j in range(len(input_data)):
    min_dist = 1000
    for k in range(len(station_list)):
        dist = calc_radial_dist(input_data.iloc[j]['PXVAL'], input_data.iloc[j]['PYVAL'], station_list.iloc[k]['LATITUDE'], station_list.iloc[k]['LONGITUDE'])
        if dist < min_dist:
            min_dist = dist
    min_dist_array.append(min_dist)
    

model_results = pd.DataFrame(columns = ['LOCATIONID', 'location_name', 'row_count', 'project_count', 'r2_test', 'rmse', 'mean_cost', 'percent_error', 'r_squared', 'f_statistic',
                                                'p_value_f', 'co_size','co_storey',
                                                'co_time','co_eplot','co_dis', 'co_time','se_size','se_storey',
                                                'se_time','se_eplot','se_dis', 'se_time', 'p_size','p_storey',
                                                'p_time','p_eplot','p_dis', 'p_time', 'condition_no'])
for i in input_data.LOCATIONID.unique():
    
    p1 = input_data[input_data['LOCATIONID']==i]
    
    
    if len(p1) > 1:
        p1['diff'] = (p1['DOS'] - p1['ENDDATE']).dt.days
      
    ##LINEAR MULTIPLE REGRESSION MODEL
    ##Assessing and essential libraries installation
    
    
        #sns.distplot(p2['TOTALCOST'])
    
        X = p1[['SALEABLE_FLAT_SIZE','STOREY', 'TOTALSUPPLY_SQFT_ASONDATE', 'diff', 'DISTANCE_KM', 'TIME_TO_REACH_MIN']]
        
        y = p1[['TOTALCOST']]
    
        X_train, X_test, y_train, y_test= train_test_split (X, y, test_size=0.3, random_state=198)
        
        lm = LinearRegression()
        
        lm.fit(X_train, y_train)
        
        #print(lm.intercept_)
        
        #lm.coef_
        
        #X.columns
        
        predictions = lm.predict(X_test)
        
        #plt.scatter(y_test, predictions)
        
        #sns.distplot(y_test-predictions)
    
        r2_test = metrics.r2_score(y_test, predictions)
    
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
        mean_cost = p1.TOTALCOST.mean()
        percent_error = rmse/mean_cost
        #print(percent_error)
        
        model = sm.OLS(y,X).fit()
        predictions = model.predict(X)
        model.summary()
            
        location_name = p1.iloc[0]['LOCATION']
        row_count = len(p1)
        project_count = len(p1['PROJECT_ID'].unique())
        r_squared = model.rsquared
        f_statistic = model.fvalue
        p_value_f = model.f_pvalue
        co_size = model.params.iloc[0]
        co_storey = model.params.iloc[1] 
        co_time = model.params.iloc[3]
        co_eplot = model.params.iloc[2]
        co_dis = model.params.iloc[4]
        co_time = model.params.iloc[5]
        se_size = model.bse.iloc[0]
        se_storey = model.bse.iloc[1] 
        se_time = model.bse.iloc[3]
        se_eplot = model.bse.iloc[2]
        se_dis = model.bse.iloc[4]
        se_time = model.bse.iloc[5]
        p_size = model.pvalues.iloc[0]
        p_storey = model.pvalues.iloc[1] 
        p_time = model.pvalues.iloc[3]
        p_eplot = model.pvalues.iloc[2]
        p_dis = model.pvalues.iloc[4]
        p_time = model.pvalues.iloc[5]
        condition_no = model.condition_number
        
        single_results = pd.DataFrame([[i, location_name, row_count, project_count, r2_test, rmse, mean_cost, percent_error, r_squared, f_statistic,
                                                p_value_f, co_size, co_storey,
                                                co_time, co_eplot, co_dis, co_time, se_size, se_storey,
                                                se_time, se_eplot, se_dis, se_time, p_size, p_storey,
                                                p_time, p_eplot, p_dis, p_time, condition_no]], columns = ['LOCATIONID', 'location_name', 'row_count', 'project_count', 'r2_test', 'rmse', 'mean_cost', 'percent_error', 'r_squared', 'f_statistic',
                                                'p_value_f', 'co_size','co_storey',
                                                'co_time','co_eplot','co_dis', 'co_time','se_size','se_storey',
                                                'se_time','se_eplot','se_dis', 'se_time', 'p_size','p_storey',
                                                'p_time','p_eplot','p_dis', 'p_time', 'condition_no'])
        
        model_results = pd.concat([model_results, single_results])
    
    
SE_dists = model.bse.iloc[4]

corr = X.corr() 
corr
sns.heatmap(corr)
np.corrcoef(X)

#r_squared
    f-statistic
    p_value_f
    co_size
    co_storey
    co_time
    co_eplot
    se_size
    se_storey
    se_time
    se_eplot
    se_size
    se_storey
    se_time
    se_eplot
    condition_no

