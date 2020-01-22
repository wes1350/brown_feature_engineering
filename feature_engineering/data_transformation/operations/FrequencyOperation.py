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

        if new_feature_cache is None:
            for i in range(len(non_duplicate_cols)):
                val_cts = df[non_duplicate_cols[i]].value_counts().to_dict()
                new_values[:, i] = df[non_duplicate_cols[i]].apply(lambda x: val_cts[x])
        else:
            for i in range(len(non_duplicate_cols)):
                name = non_duplicate_cols[i]
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                else:
                    val_cts = df[non_duplicate_cols[i]].value_counts().to_dict()
                    new_values[:, i] = df[non_duplicate_cols[i]].apply(lambda x: val_cts[x])
                    new_feature_cache[name] = new_values[:, i]

        if concatenate_originals:
            return pd.concat([df, pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                                name in non_duplicate_cols], index=df.index)], axis=1)
        else:
            return pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                                name in non_duplicate_cols], index=df.index)


def getAllFrequencyOperations():
    return [OneTermFrequencyOperation()]
