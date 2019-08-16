from .Operation import Operation
import numpy as np
import pandas as pd

class UnionOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = "union"
        self.opType = "union"

    @staticmethod
    def transform(df1, df2):
        # If they have different numbers of data points, we cannot union them
        if df1.shape[0] != df2.shape[0]:
            raise Exception("Cannot apply union operation to datasets with different numbers of rows!")

        # Determine if one has more features
        col_names_1 = set(df1.columns)
        col_names_2 = set(df2.columns)
        first_is_larger = len(col_names_1) >= len(col_names_2)

        # Determine unique columns in the smaller set
        unique_cols_in_smaller_set = col_names_2 - col_names_1 if first_is_larger else col_names_1 - col_names_2
        # If none exist, then one must be a subset of the other. Can occur if the same operation is used in different
        # branches of the same tree. e.g. root -> cos will be a subset of root -> sin -> cos even though they are separate
        if len(unique_cols_in_smaller_set) == 0:
            # print("Tried to union two dataframes, but one was a subset of the other")
            if first_is_larger:
                return df1
            else:
                return df2

        # Now we know that neither is a subset of the other, so we actually union them
        unique_cols = list(unique_cols_in_smaller_set)
        new_values = np.zeros((df1.shape[0], len(unique_cols)))  # will hold the new column values

        if first_is_larger:
            for i in range(len(unique_cols)):
                new_values[:, i] = df2[unique_cols[i]].values
            return pd.concat([df1, pd.DataFrame(new_values, columns=unique_cols, index=df1.index)], axis=1)
        else:
            for i in range(len(unique_cols)):
                new_values[:, i] = df1[unique_cols[i]].values
            return pd.concat([df2, pd.DataFrame(new_values, columns=unique_cols, index=df2.index)], axis=1)
