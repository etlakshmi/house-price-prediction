# Function to read the dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def read_csv(filename):
    try:
        df=pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Function to drop columns

def drop_columns(df,columns):
    return df.drop(columns=columns, axis=1)

# Function to encode categorical columns

def encode_categorical_columns(df,cat_cols):
    df_encoded=df.copy()
    encoders={}
    for col in cat_cols:
        le=LabelEncoder()
        df_encoded[col]=le.fit_transform(df_encoded[col].astype(str))
        encoders[col]=le
    return df_encoded, encoders

# Function to address missing values

def fill_na_with_mean(df,cols):
    df_filled=df.copy()
    mean_values={}
    for col in cols:
        mean_value=df_filled[col].mean()
        df_filled[col]=df_filled[col].fillna(mean_value)
        mean_values[col]=mean_value
    return df_filled, mean_values

# Function to split training and testing data

def split_train_test(df,target,test_size=0.2,random_state=42):
    x=df.drop(columns=[target])
    y=df[target]
    return train_test_split(x,y,test_size=test_size,random_state=random_state)

# Function to train regression models

def train_regression_models(x_train,x_test,y_train,y_test):
    results={}
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    y_pred_lr=lr.predict(x_test)
    results["LinearRegression"]={"model":lr,"R2":r2_score(y_test,y_pred_lr),"RMSE":np.sqrt(mean_squared_error(y_test,y_pred_lr))}

    rf=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)
    rf.fit(x_train,y_train)
    y_pred_rf=rf.predict(x_test)
    results["RandomForestRegressor"]={"model":rf,"R2":r2_score(y_test,y_pred_rf),"RMSE":np.sqrt(mean_squared_error(y_test,y_pred_rf))}

    return results

# Function to predict the user input

def predict_from_user_input(model,user_input,feature_columns,encoders,mean_values):
    df=pd.DataFrame([user_input])
    for col, le in encoders.items():
        if col in df.columns:
            value=df[col].astype(str)
            df[col] = value.apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    for col, mean in mean_values.items():
        if col not in df.columns:
            df[col]=mean

    df=df.reindex(columns=feature_columns,fill_value=0)

    return model.predict(df)[0]