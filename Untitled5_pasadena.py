#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:02:37 2019

@author: jboyce
"""

import pandas as pd
import pyodbc
import time


# In[2]:


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


# In[3]:


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


# In[4]:


#check for number of nulls for each column
print("Total Count of Null Records \n" + str(df_scrub.isna().count()) + "\n")


# In[5]:


#check the proportion of records that are missing
print("\n Total Percentage of Null Records \n" + str(df_scrub.isna().mean()) + "\n") #need to fix this to count or % of total using count or upb


# In[6]:


#df datatype summary
df_scrub.info()

#add categorical and non-categorical dfs
#df_scrub_cat=df_scrub[df_scrub[]]


# In[7]:


#convert to datetime and confirm converted field
df_scrub['BidDate'] = pd.to_datetime(df_scrub['BidDate'])
df_scrub.info()


# In[8]:


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
           'Last10',
           'CRTFlag',
           'ClientCRTFlag',
           'ClientId',
           'ClientName',
           'BidDate',
           'BidMonth',
           'BidWeek',
           'PropertyStateName'] #just added removal of CRTFlag and ClientID


# In[9]:


#drop features / columns not needed
df_scrub_drpd = df_scrub.drop(drop_list, axis=1)
df_scrub_drpd.head()


# In[10]:


df_scrub_drpd.shape


# In[11]:


df_scrub_drpd.columns


# In[12]:


df_scrub_drpd.index


# In[13]:


#create df of numbers and booleans
#lost index during this step
df_scrub_drpd_nums = df_scrub_drpd.select_dtypes(exclude='object') #.reset_index()
df_scrub_drpd_nums.shape


# In[14]:


#probably should delete ClientId (not a real number)
df_scrub_drpd_nums.head()


# In[15]:


#create df of categorical objects
df_scrub_drpd_cat = df_scrub_drpd.select_dtypes(include='object')
df_scrub_drpd_cat.shape


# In[16]:


#create one hot encoding for categorical features
df_scrub_onehot = pd.get_dummies(df_scrub_drpd[['PropertyTypeRM','LoanPurpose','LoanPurposeTypeName','RefinanceCashoutDeterminationTypeName','State']], prefix="", prefix_sep="")
df_scrub_onehot.head()


# In[17]:


#scale the non one hot fields (nums) and preview a couple of rows
import numpy as np
from sklearn.preprocessing import StandardScaler

X1 = df_scrub_drpd_nums.values #[:,1:]  skipped BidDate since values cannot be timestamp
X1 = np.nan_to_num(X1)
cluster_dataset = StandardScaler().fit_transform(X1)
cluster_dataset[:2]


# #### Create df of normalized numeric features

# In[18]:


#re-create the df (without bid date) to include the scaled/transformed data
df_scrub_drpd_nums_scaled = pd.DataFrame(
    cluster_dataset,
     index=list(df_scrub_drpd_nums.index),
     columns=df_scrub_drpd_nums.columns)
df_scrub_drpd_nums_scaled.head() #WORKED...NOW NEED TO ADD ONE HOT AFTER TRANSFORMED


# In[19]:


#scale the one hot fields
X_onehot = df_scrub_onehot.values
X_onehot = np.nan_to_num(X_onehot)
cluster_dataset_onehot = StandardScaler().fit_transform(X_onehot)
cluster_dataset_onehot[:2]


# #### Create df of normalized one hot features

# In[20]:


#add columns back to scaled one hot fields
#re-create the df (without bid date) to include the scaled/transformed data
df_scrub_drpd_onehot_scaled = pd.DataFrame(
    cluster_dataset_onehot,
     index=list(df_scrub_onehot.index),
     columns=df_scrub_onehot.columns)
df_scrub_drpd_onehot_scaled.head()


# In[21]:


df_scrub_drpd.filter(items=['NewCRTFLag','HB_Flag','Won_Bid']).head()


# In[22]:


#create bool df using columns that should be boolean
df_scrub_drpd_bool = df_scrub_drpd.filter(items=['NewCRTFLag','HB_Flag','Won_Bid'])
df_scrub_drpd_bool.head()


# In[23]:


#set Y/N and Yes/No values to boolean
df_scrub_drpd_bool['NewCRTFLag'] = np.where(df_scrub_drpd_bool['NewCRTFLag'] == 'Yes', 1, 0)


# In[24]:


df_scrub_drpd_bool['HB_Flag'] = np.where(df_scrub_drpd_bool['HB_Flag'] == 'Y', 1, 0)


# In[25]:


df_scrub_drpd_bool['Won_Bid'] = np.where(df_scrub_drpd_bool['Won_Bid'] == 'TRUE', 1, 0)


# In[26]:


df_scrub_drpd_bool.head()


# In[27]:


df_scrub_drpd_bool.shape


# #### Normalize the boolean columns of NewCRTFLag and HB_Flag

# In[31]:


#scale the boolean fields
X_bool = df_scrub_drpd_bool.values[:,:-1]  #skipped Won_Bid column, which is the last column and the target
X_bool = np.nan_to_num(X_bool)
cluster_dataset_bool = StandardScaler().fit_transform(X_bool)
cluster_dataset_bool[:2]


# #### Re-create df / add back column names for normalized boolean columns

# In[32]:


#add columns back to scaled one hot fields
#re-create the df (without bid date) to include the scaled/transformed data
df_scrub_drpd_bool_scaled = pd.DataFrame(
    cluster_dataset_bool,
     index=list(df_scrub_drpd_bool.index),
     columns=df_scrub_drpd_bool.columns[:-1]) #skip Won_Bid
df_scrub_drpd_bool_scaled.head()


# #### Combine/concatenate dataframes of the normalized numeric, one hot and boolean dataframes

# In[33]:


#merge num and cat df's
df_scrub_drpd_scaled = pd.concat([df_scrub_drpd_nums_scaled, df_scrub_drpd_onehot_scaled, df_scrub_drpd_bool_scaled], axis=1)
df_scrub_drpd_scaled.head()


# In[34]:


df_scrub_drpd_scaled.shape


# #### Set y as Won_Bid (keep boolean 1/0, not normalized)

# In[35]:


#set y (not normalized)
y = np.asarray(df_scrub_drpd_bool['Won_Bid'])
y [0:5]


# #### Let's define our X and y for training and testing

# In[36]:


X = df_scrub_drpd_scaled.values
X[:2]


# In[37]:


#perform logistic regression on train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# #### Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[39]:


yhat = LR.predict(X_test)
yhat


# In[40]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[41]:


from sklearn.metrics import jaccard_similarity_score, log_loss

print("Log Loss is: {}".format(log_loss(y_test, yhat_prob)))
print("Jaccard Similarity Score is: {}".format(jaccard_similarity_score(y_test, yhat)))


# #### Plot the Confusion Matrix

# In[43]:


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[44]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Win_Bid=1','Win_Bid=0'],normalize= False,  title='Confusion matrix')


# In[45]:


print("Classification Report - Logistic Regression \n \n  {}".format(classification_report(y_test, yhat)))


# ---
# ### Decision Tree
# 

# #### Modeling

# In[46]:


from sklearn.tree import DecisionTreeClassifier

dTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dTree.fit(X_train,y_train)
dTree # it shows the default parameters


# #### Prediction

# In[47]:


predTree = dTree.predict(X_test)
print (predTree [0:5])
print (y_test [0:5])


# #### Evaluation

# In[48]:


from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", round(metrics.accuracy_score(y_test, predTree),4))


# ### Create Function to iterate through max depth accuracies

# In[83]:


#iterate through max_depth to find optimal depth to minimize entropy

def dtree_max_depth_iterator():
    depth_list=[4,6,8,12]
    dtree_criterion="entropy"
    dtree_acc_score=[]
    dtree_acc_dict={}
    
    #import decision tree libraries
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    import matplotlib.pyplot as plt
    
    for i in range(0,len(depth_list)):

        #set decision tree
        dTree = DecisionTreeClassifier(criterion=dtree_criterion, max_depth = depth_list[i])
        print("i = "+str(i)) 
        #print("Length of list is: "+str(len(depth_list))) #used this to print the iteration number during testing
        
        #fit the training set
        dTree.fit(X_train,y_train)
        
        
        #prediction
        predTree = dTree.predict(X_test)
        
        
        #get the accuracy score
        dtree_acc_score.append(round(metrics.accuracy_score(y_test, predTree),6))
        dtree_acc_dict[depth_list[i]] = dtree_acc_score[i]
        print("Max Depth is: {}; Accuracy Score is: {}".format(depth_list[i], dtree_acc_dict[depth_list[i]]))
        print(dtree_acc_dict)
        print("Completed {} runs".format(i+1))
        print("\n")
        
    return dtree_acc_score
    return dtree_acc_dict
    
    
dtree_max_depth_iterator()


# <hr>
# 
# <div id="visualization">
#     <h2>Visualization</h2>
#     Lets visualize the tree
# </div>

# In[ ]:


# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[29]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


#ITERATE THROUGH MAX_DEPTHS?
dot_data = StringIO()
filename = "wonbidtree.png"
featureNames = df_scrub_drpd_scaled.columns #[0:5]
targetNames = df_scrub_drpd_bool['Won_Bid'].unique().tolist() #df_scrub_drpd_bool['Won_Bid'])
out=tree.export_graphviz(dTree,feature_names=featureNames, out_file=dot_data, class_names= str(np.unique(y_train)), filled=True,  special_characters=True,rotate=False)  #HAD TO ADD STR FOR Y_TRAIN
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# ---
# ### Support Vector Machine (SVM)
# #### Test each non-linear SVM kernel type (Polynomial, Radial based function (RBF), and Sigmoid)
# 

# ##### Radial based function (RBF)

# In[69]:


from sklearn import svm

clf = svm.SVC(kernel='rbf', gamma='auto') #added 'auto' to gamma based on warning
clf.fit(X_train, y_train)


# In[70]:


#df['dayofweek'] = df['effective_date'].dt.dayofweek


# In[55]:


#got this correlation matrix from stackoverflow (referenced in Google Sheet)
import matplotlib.pyplot as plt

f = plt.figure(figsize=(20, 17))
plt.matshow(df_scrub_drpd_scaled.corr(), fignum=f.number)
plt.xticks(range(df_scrub_drpd_scaled.shape[1]), df_scrub_drpd_scaled.columns, fontsize=14, rotation=45)
plt.yticks(range(df_scrub_drpd_scaled.shape[1]), df_scrub_drpd_scaled.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show() #added this and it displayed correlation matrix


# In[ ]:





# In[56]:


#CHECK COLUMNS IN X1 AND X_ONEHOT. REMOVE BIDWON COLUMN IF NEEDED
#create correlation matrix
#logistic regression
#show precision check


# In[ ]:




