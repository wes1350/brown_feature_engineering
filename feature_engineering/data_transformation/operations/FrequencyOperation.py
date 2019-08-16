from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd
import bisect

class FrequencyOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "frequency"

    def is_redundant_to_self(self):
        return True

class OneTermFrequencyOperation(FrequencyOperation):
    def __init__(self):
        super().__init__()
        self.operation = "one_term_frequency"

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
                current_col = df[non_duplicate_col_names_list[i].replace(self.getOperation() + " ", "", 1)]
                val_cts = current_col.value_counts().to_dict()
                new_values[:, i] = current_col.apply(lambda x: val_cts[x])
        else:
            for i in range(len(non_duplicate_col_names_list)):
                name = non_duplicate_col_names_list[i]
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                else:
                    current_col = df[non_duplicate_col_names_list[i].replace(self.getOperation() + " ", "", 1)]
                    val_cts = current_col.value_counts().to_dict()
                    new_values[:, i] = current_col.apply(lambda x: val_cts[x])
                    new_feature_cache[name] = new_values[:, i]

        return pd.concat([df, pd.DataFrame(new_values, columns=non_duplicate_col_names_list, index=df.index)], axis=1)

class TwoTermFrequencyOperation(FrequencyOperation):
    def __init__(self):
        super().__init__()
        self.operation = "two_term_frequency"

    def transform(self, df, new_feature_cache=None):
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        numeric_col_names = [name for name in numeric_col_names if "__dummy__" not in name]
        numeric_col_names_set = set(numeric_col_names)
        unique_name_pairs = []
        for i in range(len(numeric_col_names)):
            for j in range(i + 1, len(numeric_col_names)):
                old_name_1 = numeric_col_names[i]
                old_name_2 = numeric_col_names[j]
                new_name = self.getOperation() + "(" + old_name_1 + ", " + old_name_2 + ")"
                if not new_name in numeric_col_names_set:
                    unique_name_pairs.append((old_name_1, old_name_2))

        new_values = np.zeros((df.shape[0], len(unique_name_pairs)))  # will hold the new column values of df

        if new_feature_cache is None:
            pass
        else:
            pass


def getAllFrequencyOperations():
    return [OneTermFrequencyOperation()]#, TwoTermFrequencyOperation()]
