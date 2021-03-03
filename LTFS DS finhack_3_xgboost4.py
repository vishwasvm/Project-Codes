# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:15:38 2021

@author: Vishwas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.metrics import classification_report,confusion_matrix
import random 
random.seed(110)

df=pd.read_csv('V:LTFS Data Science FinHack 3/Data_csv/Train/train_Data.csv')
df.info()

# Rename column "Top-up Month"
df.rename(columns = {'Top-up Month':'Top_up_Month'}, inplace = True) 

# Remove columns 'ID', 'Area', 'AssetID', 'City' and 'ZiPCODE' 
df.drop(['Area', 'AssetID', 'City', 'State', 'DisbursalDate', 'DisbursalAmount', 'ManufacturerID', 'ZiPCODE'], axis=1, inplace=True)

# Remove NA
df=df.dropna()

# find number of duplicate rows in dataframe
print("\nDuplicate rows in DataFrame:")
print(df.duplicated().sum())

# remove/drop duplicate rows
df.drop_duplicates(subset=None, keep='first', inplace=True) 
print(df)

# Recalculate LTV
df['LTV'] = (df['AmountFinance'] / df['AssetCost']) *100

# Remove decimal of columns in data frame, convert float to int at the same time
m=(df.dtypes=='float')
df.loc[:,m]=df.loc[:,m].astype(int)

# Calculate Imbalance percentage in "Top_up_Month"
100*(df.Top_up_Month.value_counts())/ (len(df))

# remove outliers based on Interquartile Range(IQR)
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1 - 1.5 * IQR
upper_bound=Q3 + 1.5 * IQR
print(lower_bound,upper_bound)

df1 = df[~((df < lower_bound) |(df > upper_bound)).any(axis=1)]
df1.shape

# Total Interest charged on loan 
df1['Interest_on_loan']= df1['EMI']*df1['Tenure']-df1['AmountFinance']
# drop negative numbers in "Interest_on_loan"
df1 = df1.drop(df1[df1.Interest_on_loan <= 0].index)

# EMI_income_ratio
df1['loanee_emi_income_ratio'] = df1['EMI']/df1['MonthlyIncome']

# Total_loan_amount_to_pay
df1['loanee_total_loan_amount'] = df1['EMI']*df1['Tenure']

# Total_income_during_loan_period
df1['loan_period_total_income'] = df1['MonthlyIncome'] * df1['Tenure']

# Residual_income_during_loan_period
df1['loan_period_residual_income'] = df1['loan_period_total_income'] - df1['loanee_total_loan_amount']
# check for negative values
np.sum((df1.loan_period_residual_income < 0).values.ravel())
# replace negative values with 0
df1['loan_period_residual_income'][df1['loan_period_residual_income'] < 0] = 0

# loan_amount_ratio
df1["ratio_loan_amount"] = df1["AmountFinance"] / df1["Tenure"]
# income_emi_ratio
df1["ratio_income_emi"] = df1["MonthlyIncome"] / df1["EMI"]

df1=df1.dropna()
df1.sum()

# count infinity in column 'EMI_income_ratio'
c = np.isinf(df1['loanee_emi_income_ratio']).values.sum() 
print("It contains " + str(c) + " infinite values") 
# remove values with inf in column 'EMI_income_ratio'
df1 = df1.replace([np.inf, -np.inf], np.nan)
df1=df1.dropna()
# OR replace inf with 0
#from numpy import inf
#df1['loanee_emi_income_ratio'][df1['loanee_emi_income_ratio'] == inf] = 0

# Reset index: 
df1.reset_index(drop=True, inplace=True)

# log of column:
df1['log_base10'] = np.log10(df1['AmountFinance']) 
df1['Interest_log'] = np.log10(df1['Interest_on_loan']) 

# drop zero and negative numbers in "MonthlyIncome"
df1 = df1.drop(df1[df1.MonthlyIncome <= 0].index)
df1.reset_index(drop=True, inplace=True)
df1['Monthly_log'] = np.log10(df1['MonthlyIncome']) 

#Converting datetime to ordinal 
import datetime as dt
df1['AuthDate']=pd.to_datetime(df1['AuthDate'])
df1['Month'] = df1['AuthDate'].dt.month 
df1['Week'] = df1['AuthDate'].dt.week 
df1['AuthDate']=df1['AuthDate'].map(dt.datetime.toordinal)

df1['MaturityDAte']=pd.to_datetime(df1['MaturityDAte'])
df1['MaturityDAte']=df1['MaturityDAte'].map(dt.datetime.toordinal)

# Fill empty by mode 
df1=df1.fillna(df1.mode().iloc[0])
# to reset index: 
df1.reset_index(drop=True, inplace=True)

# Map PaymentMode
from collections import ChainMap
L1 = ['PDC', 'Cheque', 'PDC_E','PDC Reject']
L2 = ['ECS','ECS Reject']
L3 = ['Direct Debit','Auto Debit']
L4 = ['Billed']
L5 = ['SI Reject']
L6 = ['Escrow']
d = ChainMap(dict.fromkeys(L1, 'cat1'), dict.fromkeys(L2, 'cat2'), dict.fromkeys(L3, 'cat3'), dict.fromkeys(L4, 'cat4'), dict.fromkeys(L5, 'cat5'), dict.fromkeys(L6, 'cat6'))
# Map values
df1['PaymentMode'] = df1['PaymentMode'].map(d.get)

# Create bins for AGE
bins = [18,35,50,100]
slots = ['Low Risk','Medium Risk','High Risk']

df1['AGE']=pd.cut(df1['AGE'],bins=bins,labels=slots)

# Create bins for MonthlyIncome
bins1 = [0,25000,50000,200000,1000000,10000000]
slots1 = ['Low Income','Medium Income','High Medium Income','High Income','Very High Income']

df1['MonthlyIncome']=pd.cut(df1['MonthlyIncome'],bins=bins1,labels=slots1)

# Create bins for LTV
bins2 = [0,25,50,100,125]
slots2 = ['Low Allot','Medium Allot','High Allot','Very High Allot']

df1['LTV']=pd.cut(df1['LTV'],bins=bins2,labels=slots2)

# Create bins for EMI
bins3 = [100,15000,50000,100000,500000]
slots3 = ['Low Emi','Medium Emi','High Emi','Very High Emi']

df1['EMI']=pd.cut(df1['EMI'],bins=bins3,labels=slots3)

# create dummies for selected categorical columns excluding target column
df1 = pd.get_dummies(df1, columns=['Frequency', 'InstlmentMode', 'LoanStatus', 'PaymentMode', 'EMI', 'LTV', 'SEX', 'AGE', 'MonthlyIncome']) 

# Convert target column to numeric using labelencoder
from sklearn.preprocessing import LabelEncoder
df1['Top_up_Month'] = LabelEncoder().fit_transform(df1['Top_up_Month'])
df1['Top_up_Month'] = df1['Top_up_Month'].astype("int")

# Move"Top_up_Month" to the last of df
df1 = df1[ [ col for col in df1.columns if col != 'Top_up_Month' ] + ['Top_up_Month'] ]

# Convert all the columns as float except 'Top_up_Month'
cols = df1.columns
df1[cols[:-1]] = df1[cols[:-1]].astype("float")

df1.info()

# Separate into input and output columns
X, y = df1.iloc[:, :-1], df1.iloc[:, -1] 

# Split train and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)

# Normalization of data 
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler()
cols_to_norm = ['BranchID', 'Tenure', 'AssetCost', 'AmountFinance', 'SupplierID', 'Interest_on_loan', 'loanee_emi_income_ratio', 'loanee_total_loan_amount', 'loan_period_total_income', 'loan_period_residual_income', 'ratio_loan_amount', 'ratio_income_emi', 'log_base10', 'Monthly_log', 'Interest_log', 'Month', 'Week']
X_train[cols_to_norm] = norm.fit_transform(X_train[cols_to_norm])
X_test[cols_to_norm] = norm.transform(X_test[cols_to_norm])

# Fit Smote 
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train, y_train) 

# Model
from xgboost import XGBClassifier   
xgmodel4 = XGBClassifier(learning_rate=0.1,n_estimators=700,max_depth=4,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,objective= 'multi:softmax',num_class=7,nthread=4,scale_pos_weight=1,seed=110)
# fit
xgmodel4.fit(X_train_res, y_train_res)

# predict
xgmodel4_predict = xgmodel4.predict(X_test)

# Import test set
X_test1=pd.read_csv('V:LTFS Data Science FinHack 3/Data_csv/cleaned dataset_test/test_data_processed.csv')
X_test1.info()

# predict
xgmodel4_predict = xgmodel4.predict(X_test1)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
print('ROCAUC score:',roc_auc_score(y_test, xgmodel4_predict))
print('Accuracy score:',accuracy_score(y_test, xgmodel4_predict))
print('F1 score:',f1_score(y_test, xgmodel4_predict, average='micro'))

from sklearn.metrics import classification_report
print(classification_report(xgmodel4_predict, y_test))
print(confusion_matrix(xgmodel4_predict, y_test))

# important features
feat_imp = pd.Series(xgmodel4.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
