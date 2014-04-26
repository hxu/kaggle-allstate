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


class FillEncoderBinarizer(classes.EncoderBinarizer):
    """
    First fills in NAs in a column, then runs it through the EncoderBinarizer
    """
    def __init__(self, column, fill_value=0):
        super(FillEncoderBinarizer, self).__init__()
        self.fill_value = fill_value
        self.column = column

    def _fill_nas(self, X):
        return X[self.column].fillna(self.fill_value)

    def fit(self, X, y=None, **fit_params):
        col = self._fill_nas(X)
        return super(FillEncoderBinarizer, self).fit(col)

    def transform(self, X, y=None):
        col = self._fill_nas(X)
        return super(FillEncoderBinarizer, self).transform(col)

    def fit_transform(self, X, y=None, **fit_params):
        col = self._fill_nas(X)
        super(FillEncoderBinarizer, self).fit_transform(col)


def allfeatures_001():
    train = classes.get_train_data()
    copy = ColumnExtractor(['group_size', 'homeowner', 'car_age', 'age_oldest', 'age_youngest', 'married_couple'])
    day = DayTransformer()
    state = StateTransformer()
    car_val = FillEncoderBinarizer('car_value', 'z')
    risk_factor = FillEncoderBinarizer('risk_factor', 0)
    c_prev = FillEncoderBinarizer('C_previous', 0)
    c_dur = FillEncoderBinarizer('duration_previous', -1)

    features = FeatureUnion([
        ('copy', copy),
        ('day', day),
        ('state', state),
        ('car_val', car_val),
        ('risk_factor', risk_factor),
        ('c_prev', c_prev),
        ('c_dur', c_dur)
    ])

    pipeline = Pipeline([
        ('filter', LastShoppingPointSelector()),
        ('features', features)
    ])

    train_x = pipeline.fit_transform(train)
