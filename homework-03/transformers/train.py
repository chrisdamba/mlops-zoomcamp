import pandas as pd # working with tabular data
from sklearn.feature_extraction import DictVectorizer # Machine Learning
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error # RMSE

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    train_dicts = df[categorical].to_dict(orient='records')

    # Instantiate a dictionary vectorizer
    dv = DictVectorizer()

    # Fit the vectorizer and transform the data into a feature matrix
    X_train = dv.fit_transform(train_dicts)


    # turn the categorical columns into a list of dictionaries
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # define target variable and assign to y_val
    target = 'duration'
    y_train = df[target].values
    y_val = df[target].values

    return X_train, y_train


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'