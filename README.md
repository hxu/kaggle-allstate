kaggle-allstate
===============

Code for http://www.kaggle.com/c/allstate-purchase-prediction-challenge/data

Some possible approaches:

 - Plan combinations appear to follow a power distribution, so focus accuracy on most common combinations
 - Some plan options appear to be correlated.  Maybe use some option choices to predict others?
 - Stratified model - Build a model for each group of plans that have the same number of shopping points
 - Build a model to predict which shopping point to use as the prediction instead of predicting the plan

Some questions to explore:

 - How large is the variation in error rates for each plan feature?

   2014-04-19 07:44:21 - classes - INFO - Feature A, score 0.879660650043
   2014-04-19 07:44:21 - classes - INFO - Feature B, score 0.889536022431
   2014-04-19 07:44:22 - classes - INFO - Feature C, score 0.873382882001
   2014-04-19 07:44:22 - classes - INFO - Feature D, score 0.905060355225
   2014-04-19 07:44:22 - classes - INFO - Feature E, score 0.891247203868
   2014-04-19 07:44:22 - classes - INFO - Feature F, score 0.875887804224
   2014-04-19 07:44:22 - classes - INFO - Feature G, score 0.80510055768

   So it seems like each individual column is pretty accurate on its own, but the accuracy drops substantially
   when all the features are considered.

   G does seem to have a bit of a lower accuracy

 - Does the accuracy of the last observed plan always get higher with more shopping points?  In other words,
   how common is it that a person looks at a plan, but instead buys a plan they viewed earlier?

   Seems like the more plans that people look at, the more likely they're to choose the last plan they looked at.
   purchase at 3rd point has 49% accuracy, up to 73% accuracy for purchase at 12th point.  Slight dropoff at 13th point
   to 66%

 - How common is it for users' information to change (e.g. car value, etc.)
 - How condensed are the interactions in time?  Do interactions all tend to happen on the same day, or on multiple days?
