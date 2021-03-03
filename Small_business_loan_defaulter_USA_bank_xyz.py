# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 02:31:02 2020

@author: Vishwas
"""

#################################################################################
# Business Problem: The bank wants to predict which companies will default on their loans based on their financial information.

# Goal: To predict whether the customer will fall under default or not

#Import libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from datetime import datetime

# Read file 
df_loan = pd.read_csv("Bank_final.csv",low_memory=False) # after preprocessing and EDA

# Convert variables that are categorical in nature : RevLineCr,LowDoc,Mis-Status
# Function to map RevLineCr
class_map = {'Y': 1, 'N': 0}
df_loan['RevLineCr'] = df_loan['RevLineCr'].map(class_map)
df_loan['RevLineCr'].value_counts()  
df_loan['LowDoc'].value_counts()

# Function to map LowDoc
class_map1 = {'Y': 1, 'N': 0}
df_loan['LowDoc'] = df_loan['LowDoc'].map(class_map1) #

# function to map Mis-Status
class_map2 = {'P I F': 1, 'CHGOFF': 0}
df_loan['MIS_Status'] = df_loan['MIS_Status'].map(class_map2)
df_loan['MIS_Status'].value_counts() # 

# Drop NA
df_loan.dropna(subset=['MIS_Status'],axis=0,inplace=True)
df_loan=df_loan.fillna(df_loan.mode().iloc[0])

# # Feature Engineering
# Apply lambda function to numeric Columns, convert to float and replace ($),(,) by ('')

df_loan['DisbursementGross'] = df_loan['DisbursementGross'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(float)


df_loan['BalanceGross'] = df_loan['BalanceGross'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(float)


df_loan['SBA_Appv'] = df_loan['SBA_Appv'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(float)


df_loan['GrAppv'] = df_loan['GrAppv'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(float)


df_loan['ChgOffPrinGr'] = df_loan['ChgOffPrinGr'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(float)

# Converting datetime to ordinal 
import datetime as dt
df_loan['ChgOffDate']=pd.to_datetime(df_loan['ChgOffDate'])
df_loan['ApprovalDate']=pd.to_datetime(df_loan['ApprovalDate'])
df_loan['DisbursementDate']=pd.to_datetime(df_loan['DisbursementDate'])
df_loan['ChgOffDate']=df_loan['ChgOffDate'].map(dt.datetime.toordinal)
df_loan['ApprovalDate']=df_loan['ApprovalDate'].map(dt.datetime.toordinal)
df_loan['DisbursementDate']=df_loan['DisbursementDate'].map(dt.datetime.toordinal)

df_loan['CCSC']=np.where(df_loan['CCSC'] =='0', 0, df_loan['CCSC'])
df_loan['CCSC']=np.where(df_loan['CCSC'] > 1, 1, df_loan['CCSC'])

# Find Collineating 
df_loan.corr()

# correlation matrix
plt.figure(figsize=(13,12))
cor = df_loan.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Split data

X = df_loan[['Name', 'City', 'State', 'Zip', 'Bank', 'BankState', 'CCSC',
       'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
       'RetainedJob', 'FranchiseCode', 'UrbanRural', 'RevLineCr', 'LowDoc',
       'ChgOffDate', 'DisbursementDate', 'DisbursementGross', 'BalanceGross',
        'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']]
y = df_loan['MIS_Status']

#Find WOE and IV
from xverse.transformer import WOE
clf = WOE()
clf.fit(X, y)
Information_vale_df=clf.iv_df
columns=['Name','City', 'State', 'Zip', 'Bank', 'BankState', 'CCSC','ApprovalFY']
X=X.drop(columns=columns,axis=0)

# Feature selection
from sklearn.feature_selection import f_classif  
f_classif(X,y)
p_values = f_classif(X,y)[1]
p_values
p_values.round(3) 

#Portion of Loan Approved
X['Portion_Approved']=(X['GrAppv']-X['SBA_Appv'])/X['SBA_Appv']

#Time Taken to disburse loan
X['Time_taken']=X['DisbursementDate']-X['ApprovalDate']
from statsmodels.stats.outliers_influence import variance_inflation_factor as viff
vif=pd.DataFrame()
vif['Variables']=X.columns
vif['Vif']=[viff(X.values,i) for i in range(X.shape[1])]
#Dropping the values which are having VIF
X.drop('GrAppv',axis=1,inplace=True)
X.drop('SBA_Appv',axis=1,inplace=True)
X.drop('DisbursementDate',axis=1,inplace=True)
X.drop('ApprovalDate',axis=1,inplace=True)

#Outlier Imputation of Columns
x_f=X.copy()
def impute_outlier(ds):
   
    Q1 = ds.quantile(0.25)
    Q3 = ds.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    ds[(ds < (Q1 - 1.5 * IQR))]=ds.min(axis=0)
    ds[(ds > (Q3 + 1.5 * IQR))]=ds.max(axis=0)

  
impute_outlier(X['Term'])
impute_outlier(X['NoEmp'])
impute_outlier(X['RetainedJob'])
impute_outlier(X['DisbursementGross'])
impute_outlier(X['BalanceGross'])
impute_outlier(X['Portion_Approved'])
impute_outlier(X['Time_taken'])

# Split train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply smote
from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# Predict
y_pred_test = model.predict(x_test)

# Accuracy score 
accuracy_score(y_test, y_pred_test)

# Confusion Matrix 
cm=confusion_matrix(y_test, y_pred_test)

# Final report
finalreport = classification_report(y_test, y_pred_test,output_dict=True)
classification_df=pd.DataFrame(finalreport).transpose()

# Importing a Cross Validation Lib and Checking its Score
classification_df
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x_train,y_train,cv=10)
scores.mean()
scores.std()

# Confusion Matrix 
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("Accuracy {:0.2f}%:".format(Accuracy))

#Precision : Completness,sensitivity
Precision = tp/(tp+fp) 
print("Precision {:0.2f}".format(Precision))

#Recall : Exactness
Recall = tp/(tp+fn) 
print("Recall {:0.2f}".format(Recall))

#Specificity 
Specificity = tn/(tn+fp)
print("Specificity {:0.2f}".format(Specificity))

## Roc Curve
import scikitplot as skplt 
y_pred_proba = model.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_test, y_pred_proba)
plt.show()

####################################################################

