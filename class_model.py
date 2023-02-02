import pandas as pd
import numpy as np
import scipy as sp

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder


from matplotlib import pyplot as plt


# get the training data
df_train=pd.read_csv("train.csv")
# see what it looks like
print(df_train.head())
print(df_train.columns)

# get the test data
df_test=pd.read_csv("test.csv")


# the code below initializes some objects
impute=SimpleImputer(missing_values=np.nan,strategy="constant")

# logistic regression model
lr=LogisticRegression()

# Random Forest Model


# function that will yield a definitive dataframe for training
 
def FinalDF(df):
    def performDF(dataF):
        # columns to drop
        cols_to_drop=["Cabin","Ticket","Name"]
        dataF.drop(cols_to_drop,axis=1,inplace=True)
        print(dataF.columns)
        # select the features with objects as datatypes
        df_objects=dataF.select_dtypes(include="object")
        print("The columns of the objects are")
        print(dataF)
        # array with null values
        cols_with_na=[]
        # impute the missing values
        for cols in list(dataF.columns):
            dataF[cols].replace(" ",np.nan)
            # the block below imputes for features with numerical data type        
            if dataF[cols].isna().sum()>1 and dataF[cols].dtype in ["int64","float64"]:
                print('We have some null values')
                # get the average of the column that has null values
                average=dataF[cols].mean()
                cols_with_na.append(cols)
                # impute
                dataF[cols].replace(np.nan,average,inplace=True)

        dataF=dataF.select_dtypes(exclude="object")
        # the block below imputes for object features
        imputed_cols=pd.DataFrame(impute.fit_transform(df_objects))
        imputed_cols.columns=df_objects.columns
        print("The imputed columns are")
        print(imputed_cols)
        dataF=pd.concat([dataF,imputed_cols],axis=1,join="inner")
        # change the datatpe of all columns to be numerical
        
        print(dataF)
        # check 
        return dataF

    # get the dataset that is fit for training
    train_df=performDF(dataF=df)

    print("The Training DF")
    print(train_df)
    # logistic regression dataframe since it needs some encoding
    def getEncodedDF(df):
        ode=OrdinalEncoder()
        # select the features that have objects as their datatype
        objects_df=df.select_dtypes(include="object")
        # iterate to encode
        encodedDF=pd.DataFrame(ode.fit_transform(objects_df))
        print("The encoded DF")
        encodedDF.columns=objects_df.columns
        print(encodedDF)
        return encodedDF


    # get the object dataframe
    df_tra=train_df.select_dtypes(exclude="object")
    print("df tra")
    print(df_tra)
    # get encoded data for object datatypes
    this_df=getEncodedDF(train_df)
    # turn it to a dataframe (confirmation)
    this_df=pd.DataFrame(this_df)
    # concatenate the encoded dataframe with the dataframe with no objects
    this_df=pd.concat([this_df,df_tra],axis=1,join="inner")

    for cols in this_df.columns:
            this_df[cols].astype("float64")
    print("the train df columns are")
    print(this_df.columns)
    return this_df



this_df=FinalDF(df=df_train)
test_df=FinalDF(df=df_test)

# predictor variables dataframe
x=this_df.drop(["Survived","Sex"],axis=1)


test_df=test_df[x.columns]
print("The test Data is")
pd.DataFrame(test_df).to_csv("output.csv")

print(test_df)


# target variables dataFrame
y=this_df["Survived"]

# perform some splits on the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)


# the function below trains and tests the model accuracy
def modelStuff(n):
    
    # logistic regression
    # lr.fit(X=x,y=y)
    # predictions=lr.predict(X=test_df)
    # perform correlation in the predictions and the real value
    # corr,p_value=sp.stats.pearsonr(y_test,predictions)
    # print(f"The correlation for the Logistic Regression Model is {corr}")

    # random forests
    rf=RandomForestClassifier(n_estimators=n)
    rf.fit(X=x,y=y)
    rf_pred=rf.predict(test_df)
    rf_pred=pd.DataFrame(rf_pred)
    new_df=pd.concat([rf_pred,y_test],axis=1,join='inner')
    new_df.to_csv("output.csv")
    corr,pval=sp.stats.pearsonr(y_test,rf_pred)
    print(f"The correlation for the Random Forest Model at {n} estimators is {corr}")

    HXGB=HistGradientBoostingClassifier(n_estimators=n,learning_rate=5,early_stopping=5)
    HXGB.fit(x,y)
    preds=HXGB.predict(test_df)
    corr,pval=sp.stats.pearsonr(y_test,preds)
    print(f"The correlation for the Gradient Booster is {corr}")

    xgb=XGBClassifier(n_estimators=n,random_state=0)
    xgb.fit(X=x_train,y=y_train)
    predictions=xgb.predict(X=x_test)
    mae=mean_absolute_error(y_true=y_test,y_pred=predictions)
    print(f"The mae is {mae}")
    f_score,p_value=sp.stats.pearsonr(y_test,predictions)
    print(f"The correlation for the XGB model at {n} estimators is :",f_score)


modelStuff(500)