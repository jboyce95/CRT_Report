#!/usr/bin/env python
# coding: utf-8

# In[61]:


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:02:37 2019

@author: jboyce
"""

import pandas as pd
import pyodbc
import time


# In[62]:


#SQL settings, queries and df import

#clock starts to time how long the df import takes
start_tm = time.clock()

#set file location of sql query we want to import
sql_filename ='CRT_Bids Cumulative Tab_20191010.sql'
sql_path = r'M:\Capital Markets\Users\Johnathan Boyce\Misc\Programming\SQL'
sql_fileandpath = sql_path+"\\"+sql_filename


#connection configuration settings; connection == the connection to your database
sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER={w12pcgsql1};DATABASE=pricing_archive;UID=jboyce;Trusted_Connection=yes')

# define the opening of the filename and path of the sql file
query = open(sql_fileandpath)

#Read the sql file
df = pd.read_sql_query(query.read(),sql_conn)
sql_conn.close()


# In[63]:


#print out the run time, df shape, and df first five rows
print("\n")
print("Import Time (Using Time): " +str(time.clock()-start_tm))
print("Import Time (Using process_time): " +str(time.process_time()))
print("Import Time  (Using perf_counter): " +str(time.perf_counter()))
print("\n")
print("Top 5 rows\n" + str(df.head()) +"\n")
print("\n Number of rows: {}, Number of columns: {}".format(df.shape[0], df.shape[1]))


#analyze the df; create a scrub version of the df
df_scrub = df
print("\n Top 5 Rows for df_scrub \n" + str(df_scrub.head()) + "\n")


# In[63]:


#check for number of nulls for each column
print("Total Count of Null Records \n" + str(df_scrub.isna().count()) + "\n")


# In[63]:


#check the proportion of records that are missing
print("\n Total Count of Null Records \n" + str(df_scrub.isna().mean()) + "\n") #need to fix this to count or % of total using count or upb


# In[64]:


#df datatype summary
df_scrub.info()

#add categorical and non-categorical dfs
#df_scrub_cat=df_scrub[df_scrub[]]


# In[65]:


#convert to datetime and confirm converted field
df_scrub['BidDate'] = pd.to_datetime(df_scrub['BidDate'])
df_scrub.info()


# In[66]:


#create list of features/columns to drop
drop_list=['loannumber',
           'projectlegalstructuretypename',
           'pudindicator',
           'MortgageTypeName',
           'MtgIssuerTyp',
           'CouponRt',
           'BidVolume',
           'PricingLoanProgramName',
           'WonAmount',
           'ServicingStrip',
           'Trailing 30 Days',
           'Trailing 7 Days',
           'Today',
           'TR_7Dy_BidVolume',
           'TR_7Dy_WonAmount',
           'TR_30Dy_BidVolume',
           'TR_30Dy_WonAmount',
           'Last10']


# In[67]:


#drop features / columns not needed
df_scrub_drpd = df_scrub.drop(drop_list, axis=1)
df_scrub_drpd.head()


# In[68]:


df_scrub_drpd.shape


# In[88]:


df_scrub_drpd.index


# In[86]:


#create df of numbers and booleans
#lost index during this step
df_scrub_drpd_nums = df_scrub_drpd.select_dtypes(exclude='object') #.reset_index()
df_scrub_drpd_nums.shape


# In[87]:


df_scrub_drpd_nums.head()


# In[91]:


#df_scrub_drpd_nums.iloc[:,0].name
#df_scrub_drpd_nums.index[1]


# In[70]:


#create df of categorical objects
df_scrub_drpd_cat = df_scrub_drpd.select_dtypes(include='object')
df_scrub_drpd_cat.shape


# In[95]:


#create one hot encoding for categorical features
df_scrub_onehot = pd.get_dummies(df_scrub_drpd[['PropertyTypeRM','LoanPurpose','LoanPurposeTypeName','RefinanceCashoutDeterminationTypeName','State']], prefix="", prefix_sep="")
df_scrub_onehot.head()


# In[73]:


import numpy as np
from sklearn.preprocessing import StandardScaler

X = df_scrub_drpd_nums.values[:,1:] #test this to confirm which columns are called. confirmed from Assault column to end (skips Total Count)
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset[:2]


# In[92]:


df_bionic = pd.DataFrame(
    cluster_dataset,
     index=list(df_scrub_drpd_nums.index),
     columns=df_scrub_drpd_nums.columns)
df_bionic.head()


# In[ ]:




