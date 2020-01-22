from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd

class OneArgOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "one_arg"

    def transform(self, df, new_feature_cache=None, concatenate_originals=True):
        # Get the names of features that are numeric
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        numeric_col_names = [name for name in numeric_col_names if "__dummy__" not in name]
        numeric_col_names_set = set(numeric_col_names)
        # Store the names of columns that, once operated upon, will not produce duplicates
        # e.g. if we are doing a log operation, but we've done one once before, don't repeat log(A) but do log(log(A))
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
                new_values[:, i] = Tr.oneArgOperationWrapper(self.getOperation(), df[non_duplicate_cols[i]].values)
        else:
            for i in range(len(non_duplicate_cols)):
                name = self.getOperation() + "(" + non_duplicate_cols[i] + ")"
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                else:
                    new_values[:, i] = Tr.oneArgOperationWrapper(self.getOperation(), df[non_duplicate_cols[i]].values)
                    new_feature_cache[name] = new_values[:, i]
        if concatenate_originals:
            return pd.concat([df, pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                                name in non_duplicate_cols], index=df.index)], axis=1)
        else:
            return pd.DataFrame(new_values, columns=[self.getOperation() + "(" + name + ")" for
                                                                name in non_duplicate_cols], index=df.index)

class LogOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "log"

class SquareOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "square"

class SinOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "sin"

    def is_redundant_to_self(self):
        return True

class SqrtOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "sqrt"

class CosOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "cos"

    def is_redundant_to_self(self):
        return True

class ReciprocalOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "rc"

    def is_redundant_to_self(self):
        return True

class TanhOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "tanh"

    def is_redundant_to_self(self):
        return True

class SigmoidOperation(OneArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "sigmoid"

    def is_redundant_to_self(self):
        return True

def getAllOneArgOperations():
    return [LogOperation(), SquareOperation(), SinOperation(), SqrtOperation(), CosOperation(), ReciprocalOperation(), TanhOperation(), SigmoidOperation()]
