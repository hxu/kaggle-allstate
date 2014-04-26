"""
Trying a traditional learning framework using as many features as possible.

We'll take the features from the last shopping point and encode them one by one to create the training dataset
"""
from __future__ import division
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import classes
import numpy as np
import pandas as pd


class LastShoppingPointSelector(BaseEstimator):
    """
    Transforms a raw dataset by returning only the last non-purchase shopping point row for a customer
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        df = X[X['record_type'] == 0]
        return classes.get_last_observed_point(df)


class DayTransformer(BaseEstimator):
    """
    Uses label binarizer on the day column
    """
    def __init__(self):
        self.binarizer = LabelBinarizer()

    def fit(self, X, y=None):
        self.binarizer.fit(X['day'])
        return self

    def transform(self, X, y=None):
        return self.binarizer.transform(X['day'])

    def inverse_transform(self, Y, threshold=None):
        return self.binarizer.inverse_transform(Y, threshold)


class TimeTransformer(BaseEstimator):
    pass


class StateTransformer(BaseEstimator):
    """
    Transforms the state into binary columns
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('encode', LabelEncoder()),
            ('binarize', LabelBinarizer())
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X['state'])
        return self

    def transform(self, X, y=None):
        return self.pipeline.transform(X['state'])

    def inverse_transform(self, Y, threshold=None):
        return self.pipeline.inverse_transform(Y, threshold)


class GroupSizeTransformer(BaseEstimator):
    """
    Just copies over the group size directly
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass
