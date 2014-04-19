"""
Feature-wise training

Try building a model using only the choice of a feature, then incrementally build up
until all features are predicted.  Start with last observed plan as the base.

Can also include some features from the train set
"""
from __future__ import division
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import classes

logger = logging.getLogger('allstate')

logger.info("Loading data")
data = classes.get_train_data()
train, test = classes.train_test_split(data)

# Feature G has the lowest accuracy, so lets use all the other features to predict G
# G takes values of 1, 2, 3, and 4, we'll need to one-hot encode the response

logger.info("Transforming data")
# Transform the data into something that we can give to a learning algorithm
train_y = train.loc[train['record_type'] == 1, 'G']
train_data = train.loc[train['record_type'] == 0, ['customer_ID', 'shopping_pt', 'record_type'] + list('ABCDEFG')]
train_x = classes.get_last_observed_point(train_data)[list('ABCDEFG')]

# Responses need to be encoded to binary columns
y_encoder = OneHotEncoder()
train_y = y_encoder.fit_transform(train_y.reshape((train_y.shape[0], 1))).toarray()

# train_x is a df with columsn A-F
# Encode each column of train_x as a one-hot binary column.
f_encoder = OneHotEncoder()
est = RandomForestClassifier(n_estimators=150, verbose=3, oob_score=True)
train_x = f_encoder.fit_transform(train_x).toarray()
# OOB score is 0.93
logger.info("Training classifier")
est.fit(train_x, train_y)
