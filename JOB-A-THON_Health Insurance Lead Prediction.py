# -*- coding: utf-8 -*-
"""
Created on Sat Feb  27 19:10:55 2021

@author: Vishwas
"""
#### JOB-A-THON_Health Insurance Lead Prediction problem ####
#### Below is the solution for Insurance Lead Prediction ####


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df=pd.read_csv('V:Analytics Vidhya/JOB-A-THON_Health Insurance Lead Prediction/train_Df64byy.csv') # After preprocessing and EDA
df.info()

df.rename(columns = {'Health Indicator':'Health_Indicator'}, inplace = True) 

# check for null values
df.isnull().sum().sum()

# number of duplicate rows in dataframe
print("\nDuplicate rows in DataFrame:")
print(df.duplicated().sum())

# Replace 14+ with 15 in column "Holding_Policy_Duration"
df["Holding_Policy_Duration"].replace({"14+": "15"}, inplace=True)

# Replace null values with -1 
df['Holding_Policy_Duration'].fillna(-1,inplace=True)
df['Holding_Policy_Duration'].unique()

# Normalize Reco_Policy_Premium 
df['Reco_Policy_Premium'] = np.log1p(df['Reco_Policy_Premium'])

# Age difference 
df['age_diff'] =  (df['Upper_Age'] - df['Lower_Age'])

# Premium v/s age
df['prem_vs_lowerage'] = df['Reco_Policy_Premium']/df['Lower_Age']
df['prem_vs_agediff'] = df['Reco_Policy_Premium']/df['age_diff']

# Fillna with large negative value
df.fillna(-999,inplace=True)

# Replace inf values with large positive values
df.replace({np.inf:99999,-np.inf:99999},inplace=True)

# Reset index: 
df.reset_index(drop=True, inplace=True)

# Create dummies for selected columns 
df = pd.get_dummies(df, columns=['City_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse', 'Health_Indicator']) 

# Move "Response" to the last of df
df = df[ [ col for col in df.columns if col != 'Response' ] + ['Response'] ]

# Convert all the columns as float except 'Response'
cols = df.columns
df[cols[:-1]] = df[cols[:-1]].astype("float")

# Separate into input and output columns
X, y = df.iloc[:, :-1], df.iloc[:, -1] 

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=2)

# Normalization of data 
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler()
cols_to_norm = ['Region_Code', 'Upper_Age', 'Lower_Age', 'Holding_Policy_Duration', 'Holding_Policy_Type', 'Reco_Policy_Cat', 'Reco_Policy_Premium']
X_train[cols_to_norm] = norm.fit_transform(X_train[cols_to_norm])
X_test[cols_to_norm] = norm.transform(X_test[cols_to_norm])

# Fit smote on train
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train, y_train) 

# Fit XGBClassifier model  
from xgboost import XGBClassifier
xgmodel_1 = XGBClassifier(learning_rate=0.1,n_estimators=200,max_depth=4,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,nthread=4,scale_pos_weight=1,seed=110)
# Fit
xgmodel_1.fit(X_train_res, y_train_res)
# Predict probabilities
pred = xgmodel_1.predict_proba(X_test)
# Retrieve probabilities for the positive class
pos_probs = pred[:, 1]

# OR

# Fit LGBMClassifier model  
from lightgbm import LGBMClassifier
clf1 = LGBMClassifier(
            n_jobs=-1,
            learning_rate=0.0094,
            n_estimators=10000,
            colsample_bytree=0.94,
            subsample = 0.75,
            subsample_freq = 1,
            reg_alpha= 1.0,
            reg_lambda = 5.0,
        )

# Fit
clf1.fit(X_train_res, y_train_res)
# predict probabilities
pred1 = clf1.predict_proba(X_test)

# Retrieve probabilities for the positive class
pos_probs1 = pred[:, 1]

# Calculate roc auc
roc_auc = roc_auc_score(y_test, pos_probs1)
print('xgmodel1 ROC AUC %.3f' % roc_auc)

# Calculate roc curve of model
fpr, tpr, _ = roc_curve(y_test, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Import test set and predict
Xtest=pd.read_csv('test_processed2.csv')  # test data has been preprocessed
Xtest.info()
Xtest.astype("float").dtypes
# Predict on Xtest
pred = xgmodel_1.predict(Xtest)


ID = Xtest['ID']
submission_df_1 = pd.DataFrame({
                  "ID": ID, 
                  "Response": pred})

# Save as csv
submission_df_1.to_csv('submission_health_XG_1.csv', index=False)
