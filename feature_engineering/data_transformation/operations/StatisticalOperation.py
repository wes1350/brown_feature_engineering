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

    def transform(self, df, new_feature_cache=None):
        # Get the names of features that are numeric
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        new_numeric_col_names = [self.getOperation() + " " + name for name in numeric_col_names if
                                 "__dummy__" not in name]
        non_duplicate_col_names = set(new_numeric_col_names).difference(set(df.columns))  # get unique new columns
        new_values = np.zeros((df.shape[0], len(non_duplicate_col_names)))  # will hold the new column values of df
        # We must keep the list of new names in the original order, else we could run into errors later when
        # using multi-argument Operation types (e.g. sum)
        non_duplicate_col_names_list = [n for n in new_numeric_col_names if n in non_duplicate_col_names]

        if new_feature_cache is None:
            for i in range(len(non_duplicate_col_names_list)):
                new_values[:, i] = Tr.statisticalOperationWrapper(self.getOperation(),
                                                                  df[non_duplicate_col_names_list[i].replace(
                                                                      self.getOperation() + " ", "", 1)].values)
        else:
            for i in range(len(non_duplicate_col_names_list)):
                name = non_duplicate_col_names_list[i]
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                    # print("USING CACHE FOR NAME: ", name)
                else:
                    new_values[:, i] = Tr.statisticalOperationWrapper(self.getOperation(),
                                                                      df[non_duplicate_col_names_list[i].replace(
                                                                          self.getOperation() + " ", "", 1)].values)
                    new_feature_cache[name] = new_values[:, i]

        return pd.concat([df, pd.DataFrame(new_values, columns=non_duplicate_col_names_list, index=df.index)], axis=1)

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

