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
 - Does the accuracy of the last observed plan always get higher with more shopping points?  In other words,
   how common is it that a person looks at a plan, but instead buys a plan they viewed earlier?
 - How common is it for users' information to change (e.g. car value, etc.)
 - How condensed are the interactions in time?  Do interactions all tend to happen on the same day, or on multiple days?
