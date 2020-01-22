from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd

class StatisticalOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "statistical"

    def is_redundant_to_self(self):
        return True

    def transform(self, df, new_feature_cache=None, concatenate_originals=True):
        # Get the names of features that are numeric
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        numeric_col_names = [name for name in numeric_col_names if "__dummy__" not in name]
        numeric_col_names_set = set(numeric_col_names)
        # Store the names of columns that, once operated upon, will not produce duplicates
        non_duplicate_cols = []
        for i in range(len(numeric_col_names)):
            new_name = self.getOperation() + "(" + numeric_col_names[i] + ")"
            if new_name not in numeric_col_names_set:
                non_duplicate_cols.append(numeric_col_names[i])

        new_values = np.zeros((df.shape[0], len(non_duplicate_cols)))  # will hold the new column values of df
        # We must keep the list of new names in the original order, else we could run into errors later when
        # using multi-argument Operation types (e.g. sum)

        if new_feature_cache is None:
            for i in range(len(non_duplicate_cols)):
                new_values[:, i] = Tr.statisticalOperationWrapper(self.getOperation(), df[non_duplicate_cols[i]].values)
        else:
            for i in range(len(non_duplicate_cols)):
                name = self.getOperation() + "(" + non_duplicate_cols[i] + ")"
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                else:
                    new_values[:, i] = Tr.statisticalOperationWrapper(self.getOperation(), df[non_duplicate_cols[i]].values)
                    new_feature_cache[name] = new_values[:, i]
        if concatenate_originals:
            return pd.concat([df, pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                                    name in non_duplicate_cols], index=df.index)],
                             axis=1)
        else:
            return pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                     name in non_duplicate_cols], index=df.index)


class ZScoreOperation(StatisticalOperation):
    def __init__(self):
        super().__init__()
        self.operation = "zscore"

class MinMaxNormOperation(StatisticalOperation):
    def __init__(self):
        super().__init__()
        self.operation = "min_max_norm"

class BinningUOperation(StatisticalOperation):
    def __init__(self):
        super().__init__()
        self.operation = "binning_u"

class BinningDOperation(StatisticalOperation):
    def __init__(self):
        super().__init__()
        self.operation = "binning_d"

def getAllStatisticalOperations():
    return [ZScoreOperation(), MinMaxNormOperation(), BinningUOperation(), BinningDOperation()]

