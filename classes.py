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


def get_last_observed_plan(df, concatenate_columns=True, additional_cols=None):
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
    last_plans = pd.merge(max_pts, df, how='left')

    if concatenate_columns:
        # I think this is faster than apply
        string_cols = [last_plans[xx].astype(np.str) for xx in 'ABCDEFG']
        codes = reduce(lambda x, y: x + y, string_cols)
        last_plans['plan'] = codes
        return_cols = ['customer_ID', 'plan']
    else:
        return_cols = ['customer_ID', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Check if we need any additional columns back
    if additional_cols is not None:
        return_cols += additional_cols

    return last_plans[return_cols]


def truncate(df, p=0.3):
    """
    For a data frame of the training data, truncate each customer's series of shopping
    points to match the distribution found in the test data set.

    The test set histories are truncated. If the real process went something like:

    Quote1
    Quote2
    Quote3
    Quote4
    Purchase

    The test set could just have:

    Quote1
    Quote2

    The distribution appears to be a geometric distribution with p of 0.3 or so
    """
    purchased_plans = get_last_observed_plan(df, additional_cols=['shopping_pt'])
    # All test observations have at least 2 shopping points, so we add 1
    truncate_points = np.random.geometric(p, purchased_plans.shape[0]) + 1
    # Since last observation is purchased_plans' shopping_pt - 1
    last_observed = purchased_plans['shopping_pt'] - 1
    # So the observations then need to be truncated to the min of the truncate point or the last_observed shopping_pt
    truncate_points = pd.Series(map(min, zip(truncate_points, last_observed)))
    truncate_points.name = 'truncate'
    return pd.concat(purchased_plans['customer_ID'], truncate_points)