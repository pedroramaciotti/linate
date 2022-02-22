"""linate wrapper for ds algorithm"""

try:
    from delayedsparse import delayedsparse
except ModuleNotFoundError:
    raise ModuleNotFoundError

import pandas as pd

class CA(): 

    def __init__(self, n_components = 2):
        self.ds = delayedsparse.CA(n_components = n_components)


    def fit(self, X, y = None):
        self.ds.fit(X, y)

        # construct the results
        self.eigenvalues_ = self.ds.eigenvalues_

        self.total_inertia_ = None    # need to materialize S
        self.explained_inertia_ = None

    def row_coordinates(self):
        ca_rows = self.ds.F * 1
        df = pd.DataFrame(ca_rows)

        return df

    def column_coordinates(self):
        ca_cols = self.ds.G * 1
        df = pd.DataFrame(ca_cols)

        return df
