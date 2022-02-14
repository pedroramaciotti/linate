"""

Functionality 1: Benchmark


Input data: a pandas dataframe with short documents and its position in n-dimensional space
Input parameters: a selected dimensions and 2 "labels" that we think are dichotomized by the dimensions

Labels include selection of keywords and sentiment.


Output is the mean "precision", "recall", and "F1" resulting from fitting a logistic model,
using it to classify documents binarily, and see if this classification corresponds with labels.

We speak of "mean" because we adopt a leave-one-out strategy: fitting a logistic model for all but one sample, 
repeating for each sample.


"""