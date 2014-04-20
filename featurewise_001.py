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
import numpy as np
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

logger.info("Transforming test data")
test_y = test.loc[test['record_type'] == 1, 'G']
test_data = classes.truncate(test)
test_x = classes.get_last_observed_point(test_data)[['customer_ID'] + list('ABCDEFG')]

# Establish a baseline for what the accuracy of the last-observed on a truncated set is
last_obs = classes.concatenate_plan(test_x)[['customer_ID', 'plan']]
actuals = classes.concatenate_plan(test.loc[test['record_type'] == 1])[['customer_ID', 'plan']]
score = classes.score_df(last_obs, actuals)
scores = classes.col_score_df(classes.split_plan(last_obs), classes.split_plan(actuals))

test_y = y_encoder.transform(test_y.reshape((test_y.shape[0], 1))).toarray()
test_x = f_encoder.transform(test_x[list('ABCDEFG')]).toarray()

pred = est.predict(test_x)
# This gives what we want, but some rows don't have a prediction
# Iterate over each row of the prediction.  If the row doesn't have a 1 in some column, then set it to the last obs value
for i, row in enumerate(pred):
    if row.sum() == 0:
        pred[i, :] = test_x[i, -4:]

# Now recode into a single column, and update the last_obs
tmp = np.tile(y_encoder.active_features_, (pred.shape[0], 1))
pred = tmp[pred == 1]

last_obs['G'] = pred.astype(str)
last_obs = classes.concatenate_plan(last_obs)
classes.score_df(last_obs, actuals)
# Slightly lower accuracy on G compared to last observed
classes.col_score_df(last_obs, actuals)


#######
# What if we predict all plan components?
#######

logger.info("Transforming data")
# Transform the data into something that we can give to a learning algorithm
train_y = train.loc[train['record_type'] == 1, list('ABCDEFG')]
train_data = train.loc[train['record_type'] == 0, ['customer_ID', 'shopping_pt', 'record_type'] + list('ABCDEFG')]
train_x = classes.get_last_observed_point(train_data)[list('ABCDEFG')]

# Responses need to be encoded to binary columns
y_encoder = OneHotEncoder()
train_y = y_encoder.fit_transform(train_y).toarray()

# train_x is a df with columsn A-F
# Encode each column of train_x as a one-hot binary column.
f_encoder = OneHotEncoder()
est = RandomForestClassifier(n_estimators=150, verbose=3, oob_score=True)
train_x = f_encoder.fit_transform(train_x).toarray()
# OOB score is 0.93
logger.info("Training classifier")
est.fit(train_x, train_y)

logger.info("Transforming test data")
test_y = test.loc[test['record_type'] == 1, list('ABCDEFG')]
test_data = classes.truncate(test)
test_x = classes.get_last_observed_point(test_data)[['customer_ID'] + list('ABCDEFG')]

test_y = y_encoder.transform(test_y).toarray()
test_x = f_encoder.transform(test_x[list('ABCDEFG')]).toarray()

pred = est.predict(test_x)

# also seems to be lower than the last observed plan, 0.543 accuracy vs 0.547
