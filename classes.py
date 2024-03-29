from __future__ import division
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer,LabelEncoder

# Set up a default logger with a more informative format
logger = logging.getLogger('allstate')

if len(logger.handlers) == 0:
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


def get_test_data():
    return pd.read_csv('data/test_v2.csv')


def make_submission(df, filename):
    path = 'submissions/' + filename
    cols = ['customer_ID', 'plan']
    for c in cols:
        assert c in df, "Column {} must be in the data frame".format(c)
    df.to_csv(path, cols=cols, index=False)


def concatenate_plan(df):
    """
    Concatenates the plan columns in a data frame.  Modifies the data frame
    """
    string_cols = [df[xx].astype(np.str) for xx in 'ABCDEFG']
    codes = reduce(lambda x, y: x + y, string_cols)
    df['plan'] = codes
    return df


def split_plan(df):
    """
    Reverse of concatenate_plan.  Splits the plan column back into the letters

    Mutates the data frame
    """
    letters = 'ABCDEFG'
    for i, l in enumerate(letters):
        df[l] = df['plan'].apply(lambda x: x[i])
    return df


def get_last_observed_point(df):
    # Assumes that the index is customer_ID
    max_pts = pd.DataFrame(df.groupby('customer_ID', as_index=False)['shopping_pt'].max())
    last_plans = pd.merge(max_pts, df, how='left')
    return last_plans


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

    last_plans = get_last_observed_point(df)

    if concatenate_columns:
        # I think this is faster than apply
        last_plans = concatenate_plan(last_plans)
        return_cols = ['customer_ID', 'plan']
    else:
        return_cols = ['customer_ID', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Check if we need any additional columns back
    if additional_cols is not None:
        return_cols += additional_cols

    return last_plans[return_cols]


def get_truncation_point(df, p=0.3):
    """
    For a data frame of the training data, get the truncation point for each customer's series of shopping
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
    return pd.concat([purchased_plans['customer_ID'], truncate_points], axis=1)


def truncate(df):
    """
    Truncate a training data frame

    Truncation points are a data frame with one row for each customer_ID
    df has one row for each observation.
    """
    truncate_points = get_truncation_point(df)
    mask = pd.merge(df[['customer_ID', 'shopping_pt']], truncate_points, how='left')
    mask = mask['shopping_pt'] <= mask['truncate']
    mask.index = df.index
    return df.loc[mask]


def get_actual_plan(df):
    """
    Given a data frame of training data, get the actual plan for cross validation
    """
    purchases = df[df['record_type'] == 1]
    res = concatenate_plan(purchases)
    return res[['customer_ID', 'plan']]


def score_df(prediction, actual):
    """
    Expects two data frames with customer_ID and plan columns.

    Does a join on two data frames using customer_ID and checks if the plans are the same.
    Probably paranoid, but just in case the customer_IDs somehow get out of order
    """
    if 'customer_ID' in prediction and 'customer_ID' in actual:
        merged = pd.merge(prediction, actual, on='customer_ID', suffixes=('_p', '_a'))
        return accuracy_score(merged['plan_a'], merged['plan_p'])
    else:
        return accuracy_score(prediction['plan'], actual['plan'])


def col_score_df(prediction, actual):
    """
    Same as score_df, but expects cols A-E
    """
    cols = 'ABCDEFG'
    scores = []
    if 'customer_ID' in prediction and 'customer_ID' in actual:
        merged = pd.merge(prediction, actual, on='customer_ID', suffixes=('_p', '_a'))
        for c in cols:
            this_score = accuracy_score(merged[c + '_a'], merged[c + '_p'])
            logger.info("Feature {}, score {}".format(c, this_score))
            scores.append((c, this_score))
    else:
        for c in cols:
            this_score = accuracy_score(prediction[c], actual[c])
            logger.info("Feature {}, score {}".format(c, this_score))
            scores.append((c, this_score))
    return scores


def train_test_split(df, test_size=0.5):
    """
    train_test_split by customer_ID

    Using the default sklearn version is also slow and gets rid of indices
    """
    ids = df['customer_ID'].unique()
    cv = ShuffleSplit(ids.shape[0], test_size=test_size)
    train, test = next(iter(cv))
    train_ids = ids[train]
    test_ids = ids[test]
    return df[df['customer_ID'].isin(train_ids)], df[df['customer_ID'].isin(test_ids)]


class encode_cat():
    '''Wraps labelbinarizer and encoder together'''
    def __init__(self):
        self.LB=LabelBinarizer()
        self.LE=LabelEncoder()
        return
    def fit(self,X,y=None):
        self.LE.fit(X)
        self.LB.fit(self.LE.transform(X))
        return
    def transform(self,X,y=None):
        return(self.LB.transform(self.LE.transform(X)))
    def fit_transform(self,X,y=None):
        self.LE.fit(X)
        self.LB.fit(self.LE.transform(X))
        return(self.LB.transform(self.LE.transform(X)))
    def inverse_transform(self,X,y=None):
        return(self.LE.inverse_transform(self.LB.inverse_transform(X)))


class LabelFixerMixin(object):
    """
    Fix signatures on LabelEncoder and LabelBinarizer
    This enables the classes to be used in a Pipeline
    """
    def fit(self, X, y=None):
        return super(LabelFixerMixin, self).fit(X)

    def transform(self, X, y=None):
        return super(LabelFixerMixin, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelFixerMixin, self).fit_transform(X)


class FixLabelEncoder(LabelFixerMixin, LabelEncoder):
    pass


class FixLabelBinarizer(LabelFixerMixin, LabelBinarizer):
    pass


class EncoderBinarizer(Pipeline):
    """
    Transforms the state into binary columns

    Basically the same as the encode_cat, but I think conforms more to the
    Scikit API
    """
    def __init__(self):
        super(EncoderBinarizer, self).__init__([
            ('encode', FixLabelEncoder()),
            ('binarize', FixLabelBinarizer())
        ])


class MultiColLabelBinarizer(BaseEstimator, TransformerMixin):
    """
    LabelBinarizer only works with a single column at a time.  This class does LabelBinarization on
    multiple columns in a data frame and concatenates the results

    Also, OneHotEncoder doesn't seem to easily do reverse transformations
    """
    def __init__(self, neg_label=0, pos_label=1):
        self.neg_label = neg_label
        self.pos_label = pos_label

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("Only works with DataFrames.  Got {}".format(X.__class__))

        self.binarizers_ = []
        for col in X.columns:
            binarizer = LabelBinarizer(self.neg_label, self.pos_label)
            binarizer.fit(X[col].values)
            self.binarizers_.append((col, binarizer))
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("Only works with DataFrames.  Got {}".format(X.__class__))

        res = []
        for col, binarizer in self.binarizers_:
            res.append(binarizer.transform(X[col].values))

        return np.hstack(res)

    def inverse_transform(self, y, threshold=None):
        start = 0
        res = []
        cols = []
        for col, binarizer in self.binarizers_:
            n_cols = len(binarizer.classes_) if len(binarizer.classes_) > 2 else 1
            this_r = binarizer.inverse_transform(y[:, start:start+n_cols]).reshape((y.shape[0], 1))
            res.append(this_r)
            cols.append(col)
            start += n_cols

        return pd.DataFrame(np.hstack(res), columns=cols)
