"""
Trying a traditional learning framework using as many features as possible.

We'll take the features from the last shopping point and encode them one by one to create the training dataset
"""
from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import classes
import numpy as np
import pandas as pd


class LastShoppingPointSelector(BaseEstimator, TransformerMixin):
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


class DayTransformer(LabelBinarizer):
    """
    Uses label binarizer on the day column only
    """
    def fit(self, X):
        return super(DayTransformer, self).fit(X['day'])

    def transform(self, X):
        return super(DayTransformer, self).transform(X['day'])

    def fit_transform(self, X, y=None, **fit_params):
        return super(DayTransformer, self).fit_transform(X['day'])


class StateTransformer(classes.EncoderBinarizer):
    """
    Transforms the state into binary columns
    """
    def fit(self, X, y=None):
        return super(StateTransformer, self).fit(X['state'])

    def transform(self, X, y=None):
        return super(StateTransformer, self).transform(X['state'])

    def fit_transform(self, X, y=None, **fit_params):
        return super(StateTransformer, self).fit_transform(X['state'])


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """
    Copies over a single column or multiple columns of the
    input data frame without any transformations.

    Arguments:
    ----------
    columns: string or list of strings
        The columns in the input data frame to extract

    Returns:
    --------
    ndarray of shape (n_rows, len(columns))
    """

    def __init__(self, columns):
        if isinstance(columns, basestring):
            columns = [columns]
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X[self.columns]
        if isinstance(res, pd.DataFrame):
            # No reshape, just return underlying ndarray
            return res.values
        elif isinstance(res, pd.Series):
            return res.reshape((X.shape[0], 1))


class RiskFactorEncoder(classes.EncoderBinarizer):
    """
    Handles the risk factor column.  Takes values of 1-4 and NA.  We'll encode NA as 0, then run it through
    the EncoderBinarizer transformer
    """
    def __init__(self, fill_value=0):
        super(RiskFactorEncoder, self).__init__()
        self.fill_value = 0

    def fit(self, X, y=None):
        col = X['risk_factor'].fillna(self.fill_value)
        return super(RiskFactorEncoder, self).fit(col)

    def transform(self, X, y=None):
        col = X['risk_factor'].fillna(self.fill_value)
        return super(RiskFactorEncoder, self).fit(col)

    def fit_transform(self, X, y=None, **fit_params):
        col = X['risk_factor'].fillna(self.fill_value)
        return super(RiskFactorEncoder, self).fit_transform(col)
