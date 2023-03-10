import pandas as pd
import scipy as sp
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression

# training data
df=pd.read_csv("train.csv")

# test data
test_df=pd.read_csv("test.csv")
x_test=test_df.copy()

# the function below performs all that entails with feature selection such as data cleaning and preprocessing
def dfPerform(df,type):
    # drop some columns that would not lead to people dying
    cols_to_drop=["Cabin","Ticket","Name"]
    df.drop(cols_to_drop,axis=1,inplace=True)
    # perform ordinal encoding on non numerical features
    df_objects=df.select_dtypes(include="object")
    # drop the object features and remain with a df with numerical features only
    df.drop(df_objects.columns,axis=1,inplace=True)
    col_names=df_objects.columns
    # instantiate the ordinal encoder
    enc=OrdinalEncoder()
    df_objects=pd.DataFrame(enc.fit_transform(df_objects))
    # add back the column name to the encoded dataframe
    df_objects.columns=col_names
    # concatenate the encoded dataframe and the original numeric dataframe
    df=pd.concat([df,df_objects],axis=1,join="inner")
    # array for columns with null values
    cols_withNa=[]
    # instantiate the SimpleImputer to replace the null values
    impute=SimpleImputer(strategy="median",fill_value="median")
    
    for col in df.columns:
        average=df[col].mean()
        # check if a column is null
        if df[col].isna().sum() >1 and df[col].dtype in ["int64","float64"]:
            # append the column with null values
            cols_withNa.append(col)
            df[col].replace(np.nan,average)
    imputed_cols=pd.DataFrame(impute.fit_transform(df[cols_withNa]))
    imputed_cols.columns=cols_withNa

    # drop columns with null values  
    df.drop(cols_withNa,axis=1,inplace=True)
    # concatenate the imputed columns with the original dataframe
    df=pd.concat([df,imputed_cols],axis=1)
    if(type == "train"):
        for kol in df.columns:
            # perform correlation computation to get whether some certain features would really affect the survival rate
            stats,pvalue=sp.stats.pearsonr(df["Survived"],df[kol])
            print(f"{kol} has a pvalue of {pvalue}")
            # drop columns that have a Pvalue greater than 0.1
            if (pvalue >0.1):
                df.drop([kol],axis=1,inplace=True)
    return df

# separate the target and predictor variables
real_df=dfPerform(df=df,type='train')
y=real_df["Survived"]
x=real_df.drop(["Survived"],axis=1)

print(f"Training data columns is {x.columns}")

# update the test columns to match the training columns
x_testy= dfPerform(x_test,type="test")



# split the data
x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)


x_testz=x_testy[x_valid.columns]

for col in x_testz.columns:
    if x_testz[col].isna().sum()>1:
        print("We have some null values")
    else:
        print("We do not have null values")

# now we have the comprehensive dataset that we would use to train our ensembling model
# instantiate the different models


def models(mode):
    mode.fit(x,y)
    # get some predictions
    predictions=mode.predict(x_testz.head(179))
    predictions=pd.DataFrame(predictions,columns=["Survived"])
    new_preds=predictions["Survived"].apply(lambda x: 1 if x >=0.5 else 0)
    corr=new_preds.corr(y_valid,method="pearson")
    print(f"The Correlation is {corr}")
    print(new_preds)
    print(y_valid)
    coore,pval=sp.stats.pearsonr(new_preds,y_valid)
    print(coore)
    return coore
# lets start with XGBOOST
xgb=XGBRegressor(n_estimators=100)

XGBoost=models(xgb)

print(f" the XGBOOST correlation is{XGBoost}")

# lets try it with RandomForests
# rf=RandomForestRegressor(n_estimators=200,random_state=0)

# rf.fit(x,y)
# predictions=rf.predict(x_testz.head(179))
# predictions=pd.DataFrame(predictions,columns=["Survived"])
# new_preds=predictions["Survived"].apply(lambda x: 1 if x >=0.5 else 0)
# corr=new_preds.corr(y_valid,method="pearson")
# print(f"The Correlation is {corr}")
# print(new_preds)
# print(y_valid)
# coore,pval=sp.stats.pearsonr(new_preds,y_valid)
# print(coore)

# print(f"The RandomForest correlation is {}")

# lets try with logistic regression
x_testm=x_testy[x_valid.columns]
for colo in x_testm.columns:
    x_testm.replace(" ",np.nan)
    if x_testm[colo].isna().sum()>1:
        print("me ni fala")
print(x_valid)
lr=LogisticRegression(random_state=0)

lr.fit(x,y)

preds= lr.predict(x_testm)
Logist=sp.stats.pearsonr(preds,y_valid)
print(f"The Logistic Regression correlation is {Logist}")


# store both models in an array
model_Arr=[xgb]

def getScores(model):
    # now get the various errors using cross validation scores
    scores=-1 * cross_val_score(model,x,y,cv=5,scoring="neg_mean_absolute_error")
    return scores

for mode in model_Arr:
    thiScore=getScores(mode)
    print(f"The mean absolute error is of the {mode} is {thiScore.mean()}")


