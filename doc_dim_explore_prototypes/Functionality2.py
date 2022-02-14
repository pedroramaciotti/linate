"""

Functionality 2: Dimension discovery


Input data: a pandas dataframe with short documents and its position in n-dimensional space
Input parameters: 2 "labels" that we think are dichotomized by the dimensions

Labels include selection of keywords and sentiment.


Output is a direction in space, a vector in n-dimensional space, that best dichotomizes the selected labels.
Output also includes metrics, "precision", "recall", and "F1" of how this found direction dichotomizes the labels.


"""