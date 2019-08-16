from .Operation import Operation
import numpy as np
# import TransformOperations as Tr
# import pandas as pd

class CompactOneHotOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = "compact_one_hot"
        self.opType = "compact_one_hot"

    def is_redundant_to_self(self):
        return True

    @staticmethod
    def transform(df):
        categorical_dummy_cols = [col for col in df.columns.values.tolist() if "__dummy__" in col]  # just dummies
        if len(categorical_dummy_cols) != 0:  # Can happen if we feature select all the categorical variables out.
            dummy_col_prefixes = [] # Holds the prefixes of the dummy cols, i.e. the name of the categorical variable
            for c in categorical_dummy_cols:
                if c.split("__dummy__")[0] not in dummy_col_prefixes:
                    dummy_col_prefixes.append(c.split("__dummy__")[0])
            for cat_var_name in dummy_col_prefixes:
                relevant_dummies = [x for x in categorical_dummy_cols if x.split("__dummy__")[0] == cat_var_name]
                cols_to_remove = []

                # See the "size" of each dummy, i.e. # of positive entries
                dummy_positive_sizes = {}
                for c in relevant_dummies:
                    dummy_positive_sizes[c] = len(np.flatnonzero(df[c]))

                # Find out which ones are too small
                sparse_dummy_col = np.zeros(df.shape[0])
                threshold_factor = 1.0/(len(relevant_dummies)**1.5)
                for c in dummy_positive_sizes:
                    # print("->>", cat_var_name, dummy_positive_sizes[c], df.shape[0]*threshold_factor)
                    if dummy_positive_sizes[c] < df.shape[0]*threshold_factor:
                        sparse_dummy_col = np.add(sparse_dummy_col, df[c])
                        cols_to_remove.append(c)

                # Don't do anything if we aren't combining anything
                if len(cols_to_remove) > 1:
                    # print("\n\n")
                    # print(cols_to_remove)
                    # print("\n\n")
                    res = df.drop(cols_to_remove, axis=1)
                    res[cat_var_name + "__dummy__misc_combined_values"] = sparse_dummy_col
                    return res
                else:
                    return df
        else:
            return df
