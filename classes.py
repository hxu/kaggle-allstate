from __future__ import division
import logging
import pandas as pd
import numpy as np


# Set up a default logger with a more informative format
logger = logging.getLogger('allstate')
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

# Log to file, particularly useful for spot instances, since the
# log will persist even if the instance is killed (if you use a separate EBS volume)
logfile = logging.FileHandler('run.log')
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(log_formatter)
# Log to console
logstream = logging.StreamHandler()
logstream.setLevel(logging.INFO)
logstream.setFormatter(log_formatter)

logger.addHandler(logfile)
logger.addHandler(logstream)


def get_train_data():
    """
    Returns the training data
    """
    return pd.read_csv('data/train.csv')


def get_last_observed_plan(df, concatenate_columns=True):
    """
    Given a data frame with at least columns customer_ID, shopping_pt, and A-G,
    return a data frame with unique customer IDs and the final set of A-G values.

    If concatenate_columns is True, then it collapses A-G to a single column, 'plan'
    Otherwise it returns the values as individual columns

    This doesn't rely on the record_type column == 1, since that's only present on the training set
    """
    cols = ['customer_ID', 'shopping_pt', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    for c in cols:
        assert c in df, "Column {} not found in data frame".format(c)

    # Assumes that the index is customer_ID
    max_pts = pd.DataFrame(df.groupby('customer_ID', as_index=False)['shopping_pt'].max())
    last_plans = pd.merge(max_pts, df, how='left')[cols].drop('shopping_pt', 1)
    if concatenate_columns:
        string_cols = [last_plans[xx].astype(np.str) for xx in 'ABCDEFG']
        codes = reduce(lambda x, y: x + y, string_cols)
        last_plans['plan'] = codes
        last_plans = last_plans[['customer_ID', 'plan']]
    return last_plans
