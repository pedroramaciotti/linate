"""

Functionality 2: Find dichotomized labels


Input data: a pandas dataframe with short documents and its position in n-dimensional space
Input parameters: a dimension or direction (a n-dimensional vector)


Output is a set of two labels (or maybe just keywords, for simplicity?) that are well dichotomized by the direction.
Maybe the result could include the top K pairs of keywords that are best dichotomized.
Labels include selection of keywords and sentiment.

Output also includes metrics, "precision", "recall", and "F1" of how this found direction dichotomizes the labels.


"""