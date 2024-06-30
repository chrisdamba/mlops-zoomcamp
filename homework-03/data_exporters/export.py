import mlflow
import pandas as pd # working with tabular data
from sklearn.feature_extraction import DictVectorizer # Machine Learning
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error # RMSE

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    with mlflow.start_run():
        mlflow.set_tag("developer", "chrisdamba")

        X_train, y_train = data
            # instantiate & fit our model to the TRAINING set
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        # Print the intercept
        print("Intercept:", lr.intercept_)
        mlflow.log_metric("intercept", lr.intercept_)
        mlflow.sklearn.log_model(lr, "models_mlflow")

