"""
Last quoted plan benchmark
"""
import classes
from classes import logger

train = classes.get_train_data()
actuals = classes.get_actual_plan(train)

scores = []
# Score seems to be a bit high on training, about .547-.548
for n in range(5):
    truncated = classes.truncate(train)
    prediction = classes.get_last_observed_plan(truncated)
    score = classes.score_df(prediction, actuals)
    scores.append(score)
    logger.info("Run {}, score: {}".format(n+1, score))

test = classes.get_test_data()
pred = classes.get_last_observed_plan(test)

classes.make_submission(pred, 'benchmark_001.csv')