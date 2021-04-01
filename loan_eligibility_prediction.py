
##### Loan prediction problem, datascience competition organized by analyticsvidhya.com

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import random
SEED = 110

random.seed(SEED)
np.random.seed(SEED)

# Load train and test data
df = pd.read_csv('train_ctrUa4K.csv')
test_df = pd.read_csv('test_lAUu6dG.csv')
df.info()
test_df.info()

# Remove column 'ID'
df.drop(['Loan_ID'], axis=1, inplace=True)

# Check for null values
df.isnull().sum().sum()
df.isnull().sum() / len(df)

test_df.isnull().sum().sum()
test_df.isnull().sum() / len(df)

# Number of duplicate rows in dataframe
print("\nDuplicate rows in DataFrame:")
print(df.duplicated().sum())

print("\nDuplicate rows in DataFrame:")
print(test_df.duplicated().sum())

# Imputing Gender
df['Gender'].unique()

sns.countplot(df['Gender'])

most_freq = df['Gender'].mode()[0]
print(most_freq)

most_freq_1 = test_df['Gender'].mode()[0]
print(most_freq_1)

df['Gender'] = df['Gender'].fillna(most_freq)
test_df['Gender'] = test_df['Gender'].fillna(most_freq_1)

# Imputing Married
df['Married'].unique()

sns.countplot(df['Married'])

most_freq_2 = df['Married'].mode()[0]
print(most_freq_2)

df['Married'] = df['Married'].fillna(most_freq_2)

# Imputing Dependents
df['Dependents'].unique()

sns.countplot(df['Dependents'])

most_freq_3 = df['Dependents'].mode()[0]
print(most_freq_3)

most_freq_4 = test_df['Dependents'].mode()[0]
print(most_freq_4)

df['Dependents'] = df['Dependents'].fillna(most_freq_3)
test_df['Dependents'] = test_df['Dependents'].fillna(most_freq_4)

# Imputing Self_Employed
df['Self_Employed'].unique()

sns.countplot(df['Self_Employed'])

most_freq_5 = df['Self_Employed'].mode()[0]
print(most_freq_5)

most_freq_6 = test_df['Self_Employed'].mode()[0]
print(most_freq_6)

df['Self_Employed'] = df['Self_Employed'].fillna(most_freq_5)
test_df['Self_Employed'] = test_df['Self_Employed'].fillna(most_freq_6)

# Imputing LoanAmount
df['LoanAmount'].unique()

sns.countplot(df['LoanAmount'])

most_freq_7 = df['LoanAmount'].mode()[0]
print(most_freq_7)

most_freq_8 = test_df['LoanAmount'].mode()[0]
print(most_freq_8)

df['LoanAmount'] = df['LoanAmount'].fillna(most_freq_7)
test_df['LoanAmount'] = test_df['LoanAmount'].fillna(most_freq_8)

# Imputing Loan_Amount_Term
df['Loan_Amount_Term'].unique()

sns.countplot(df['Loan_Amount_Term'])

most_freq_9 = df['Loan_Amount_Term'].mode()[0]
print(most_freq_9)

most_freq_10 = test_df['Loan_Amount_Term'].mode()[0]
print(most_freq_10)

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(most_freq_9)
test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(most_freq_10)

# Imputing Credit_History
df['Credit_History'].unique()

sns.countplot(df['Credit_History'])

most_freq_11 = df['Credit_History'].mode()[0]
print(most_freq_11)

most_freq_12 = test_df['Credit_History'].mode()[0]
print(most_freq_12)

df['Credit_History'] = df['Credit_History'].fillna(most_freq_11)
test_df['Credit_History'] = test_df['Credit_History'].fillna(most_freq_12)

# Check for null values
df.isnull().sum().sum()
df.isnull().sum() / len(df)

test_df.isnull().sum().sum()
test_df.isnull().sum() / len(df)

# Number of duplicate rows in dataframe
print("\nDuplicate rows in DataFrame:")
print(df.duplicated().sum())

print("\nDuplicate rows in DataFrame:")
print(test_df.duplicated().sum())

# Loan_Status: Y=0.687296, N=0.312704
df['Loan_Status'].unique()
sns.countplot(df['Loan_Status'])
df.Loan_Status.value_counts(normalize=True)

# Replace 3+ with 4 in column "Dependents"
df["Dependents"].replace({"3+": "4"}, inplace=True)
test_df["Dependents"].replace({"3+": "4"}, inplace=True)

# Create new features 
df["ApplicantIncome_by_loanAmt"] = df["ApplicantIncome"] / df["LoanAmount"]
df["ratio_loan_amount"] = df["LoanAmount"] / df["Loan_Amount_Term"]
df["Total_income_app_co"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Total_income_by_loanAmt"] = df["Total_income_app_co"] / df["LoanAmount"]
test_df["ApplicantIncome_by_loanAmt"] = test_df["ApplicantIncome"] / test_df["LoanAmount"]
test_df["ratio_loan_amount"] = test_df["LoanAmount"] / test_df["Loan_Amount_Term"]
test_df["Total_income_app_co"] = test_df["ApplicantIncome"] + df["CoapplicantIncome"]
test_df["Total_income_by_loanAmt"] = test_df["Total_income_app_co"] / test_df["LoanAmount"]

# Log of column:
df['log_base10'] = np.log10(df['LoanAmount']) 
df['Interest_log'] = np.log10(df['ApplicantIncome']) 
test_df['log_base10'] = np.log10(test_df['LoanAmount']) 
test_df['Interest_log'] = np.log10(test_df['ApplicantIncome']) 

# Find corelation in df
corr = df.corr()
sns.heatmap(df=corr.dropna(),cmap='Blues',linewidth=0.5)
sns.heatmap(df=corr.dropna(),mask = corr < 0.8,cmap='Blues',linewidth=0.5)

# Drop columns having very high correlation: "Total_income_app_co", "Total_income_by_loanAmt"
df.drop(['Total_income_app_co'], axis=1, inplace=True)
df.drop(['Total_income_by_loanAmt'], axis=1, inplace=True)
test_df.drop(["Total_income_app_co"], axis=1, inplace=True)
test_df.drop(["Total_income_by_loanAmt"], axis=1, inplace=True)

# Create dummies for categorical columns 
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']) 
#test_df = = pd.get_dummies(df, columns=['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']) 

# Convert target column to numeric using labelencoder
from sklearn.preprocessing import LabelEncoder
df['Loan_Status'] = LabelEncoder().fit_transform(df['Loan_Status'])
df['Loan_Status'] = df['Loan_Status'].astype("int")

# Move"Loan_Status" to the last of df
df = df[ [ col for col in df.columns if col != 'Loan_Status' ] + ['Loan_Status'] ]

# Convert all the columns as float except 'Top_up_Month'
cols = df.columns
df[cols[:-1]] = df[cols[:-1]].astype("float")

# Save test dataframe as csv
test_df.to_csv(r'test_cleaned1.csv', index=False) 
df.to_csv(r'train_final2.csv', index=False) 

# Load train data
df = pd.read_csv('train_final2.csv')

# Separate into input and output columns
X, y = df.iloc[:, :-1], df.iloc[:, -1] 

# Split train and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110, stratify=y, shuffle=True)

# Normalization of data 
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler()
cols_to_norm = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome_by_loanAmt','ratio_loan_amount']
X_train[cols_to_norm] = norm.fit_transform(X_train[cols_to_norm])
X_test[cols_to_norm] = norm.transform(X_test[cols_to_norm])

# Fit Smote 
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train, y_train) 

# Model
from xgboost import XGBClassifier   
xgmodel_loan_1 = XGBClassifier(learning_rate=0.1,n_estimators=700,max_depth=4,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,nthread=4,scale_pos_weight=1,seed=110)
# fit
xgmodel_loan_1.fit(X_train_res, y_train_res)
# predict
xgmodel_loan_1_predict = xgmodel_loan_1.predict(X_test)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
print('ROCAUC score:',roc_auc_score(y_test, xgmodel_loan_1_predict))
print('Accuracy score:',accuracy_score(y_test, xgmodel_loan_1_predict))
print('F1 score:',f1_score(y_test, xgmodel_loan_1_predict, average='micro'))

from sklearn.metrics import classification_report
print(classification_report(xgmodel_loan_1_predict, y_test))
print(confusion_matrix(xgmodel_loan_1_predict, y_test))

# Predict on test data
df1 = pd.read_csv('test_cleaned1.csv')

df2 = df1[df1.columns.difference(['Loan_ID'])]
print(df2)

# Map columns in the order of train data
df2 = df2[['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Education','Gender','Married','Property_Area','Self_Employed','ApplicantIncome_by_loanAmt','ratio_loan_amount','Total_income_app_co','Total_income_by_loanAmt']]

# create dummies for selected categorical columns excluding target column
df2 = pd.get_dummies(df2, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']) 

# Normalization of data 
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler()
cols_to_norm = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','ApplicantIncome_by_loanAmt','ratio_loan_amount']
df2[cols_to_norm] = norm.fit_transform(df2[cols_to_norm])

# Predict
ypred_test = rfc11.predict(df2)

# Save prediction as csv
Loan_ID = df1['Loan_ID']
submission_df_1 = pd.DataFrame({
                  "Loan_ID": Loan_ID, 
                  "Loan_Status": ypred_test})


# Map: 'Y' and 'N'
submission_df_1['abc'] = submission_df_1['Loan_Status'].map({True: 'Y', False: 'N'})
# Remove column: 'Loan_Status' 
submission_df_1.drop(['Loan_Status'], axis=1, inplace=True)
# Rename column: 'abc'
submission_df_1.rename(columns = {'abc':'Loan_Status'}, inplace = True) 
# Save as csv
submission_df_1.to_csv(r'submission_1.csv', index=False) 
