# in case this is run outside of conda environment with python2
import mlflow
import argparse
import sys
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
import mlflow.tensorflow
# Data analysis library
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Machine Learning library
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, plot_confusion_matrix, plot_roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Model experimentation library
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Plotting library
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
warnings.filterwarnings("ignore")

# Load train loan dataset
def load_data():
    data = None
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="max")
        print("The dataset has {} samples with {} features.".format(*data.shape))
    except:
        print("The dataset could not be loaded. Is the dataset missing?")

    print(data.tail(5))
    return data

def plot_data(df1, df2, name):
    df1.rename("Ground", inplace = True)
    df2.rename("Forecast", inplace= True)
    ax = df1.plot(legend=True)
    df2.plot(ax=ax, legend=True)
    plt.grid()
    plt.title('Naive Forecasting')
    # plt.show()
    filename = f'images/{name}.png'
    plt.savefig(filename)

def spit_data(df):
    df = df["Close"]
    time_split = 2479
    time_train = df[:time_split]
    x_train = df[:time_split]
    
    time_valid = df[time_split:]
    x_valid = df[time_split:]

    print(f"\nLength of full dataset: {len(df)}")
    print(f"Length of x_valid: {len(x_valid)}")
    print(f"Length of x_train: {len(x_train)}")
    return x_train, x_valid, df

def set_experiment():
    experiment_name = "Time series forecasting"

    # Initialize MLflow client
    client = MlflowClient()

    # If experiment doesn't exist then it will create new
    # else it will take the experiment id and will use to to run the experiments
    try:
        # Create experiment 
        experiment_id = client.create_experiment(experiment_name)
    except:
        # Get the experiment id if it already exists
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id

def naive_forcast_model(df, split_time=2479):
    naive_forecast = df[split_time - 1:-1]

    return naive_forecast

def naive_experiment(experiment_id, df):

    with mlflow.start_run(experiment_id=experiment_id, run_name='naive_model') as run:
        # Get run id 
        run_id = run.info.run_uuid
        mlflow.sklearn.autolog()

        # Provide brief notes about the run
        MlflowClient().set_tag(run_id,
                            "mlflow.note.content",
                            "This is experiment for exploring different machine learning models for time series forecasting")

            
        # Define tag
        tags = {"Application": "Time series forecasting",
                "release.candidate": "Tutorial",
                "release.version": "0.0.1"}
                
        # Set Tag
        mlflow.set_tags(tags)
        
        # Log python environment details
        mlflow.log_artifact('conda.yaml')

        # Define model parameter
        params = {
            "prediction_steps": 7,
            "seed": 42,
        }
            
        # logging params
        mlflow.log_params(params)
    
        # Plot and save metrics details    
        naive_forecast = naive_forcast_model(df)
        plot_data(x_valid ,naive_forecast, "naive_forecastplot")
        filename = f'images/naive_forecastplot.png'
        # log model artifacts
        mlflow.log_artifact(filename)
        
        # Loss   
        mae = mean_absolute_error(x_valid, naive_forecast)
        mlflow.log_metrics({"Mean Absolute Error": mae})

if __name__ == "__main__":
    data = load_data()
    x_train, x_valid, df = spit_data(data)

    experiment_id = set_experiment()


    # naive_forcast = naive_forcast_model(df)
    naive_experiment(experiment_id, df)
    print("\n Process naive forcast data")
    # print(naive_forcast)
    # print(x_valid)
    # plot_data(x_valid ,naive_forcast)









