from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd

class TwoArgOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "two_arg"

    def isSymmetric(self):
        return None

    def transform(self, df, new_feature_cache=None):
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        numeric_col_names = [name for name in numeric_col_names if "__dummy__" not in name]
        numeric_col_names_set = set(numeric_col_names)
        unique_name_pairs = []

        if self.isSymmetric():
            for i in range(len(numeric_col_names)):
                for j in range(i + 1, len(numeric_col_names)):
                    old_name_1 = numeric_col_names[i]
                    old_name_2 = numeric_col_names[j]
                    new_name = self.getOperation() + "(" + old_name_1 + ", " + old_name_2 + ")"
                    if not new_name in numeric_col_names_set:
                        unique_name_pairs.append((old_name_1, old_name_2))
        else:
            for i in range(len(numeric_col_names)):
                for j in list(range(i)) + list(range(i + 1, len(numeric_col_names))):
                    old_name_1 = numeric_col_names[i]
                    old_name_2 = numeric_col_names[j]
                    new_name = self.getOperation() + "(" + old_name_1 + ", " + old_name_2 + ")"
                    if not new_name in numeric_col_names_set:
                        unique_name_pairs.append((old_name_1, old_name_2))

        new_values = np.zeros((df.shape[0], len(unique_name_pairs)))  # will hold the new column values of df

        if new_feature_cache is None:
            for i in range(len(unique_name_pairs)):
                new_values[:, i] = Tr.twoArgOperationWrapper(self.getOperation(), df[unique_name_pairs[i][0]].values,
                                                             df[unique_name_pairs[i][1]].values)
        else:
            for i in range(len(unique_name_pairs)):
                name = self.getOperation() + "(" + unique_name_pairs[i][0] + ", " + unique_name_pairs[i][1] + ")"
                if name in new_feature_cache:
                    new_values[:, i] = new_feature_cache[name]
                    # print("USING CACHE FOR NAME: ", name)
                else:
                    new_values[:, i] = Tr.twoArgOperationWrapper(self.getOperation(),df[unique_name_pairs[i][0]].values,
                                                                 df[unique_name_pairs[i][1]].values)
                    new_feature_cache[name] = new_values[:, i]

        return pd.concat([df, pd.DataFrame(new_values,
                                           columns=[self.getOperation() + "(" + p[0] + ", " + p[1] + ")" for p in
                                                    unique_name_pairs], index=df.index)], axis=1)

class SumOperation(TwoArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "sum"

    def isSymmetric(self):
        return True

class SubtractOperation(TwoArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "subtract"

    def isSymmetric(self):
        return True

class MultiplyOperation(TwoArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "multiply"

    def isSymmetric(self):
        return True

class DivideOperation(TwoArgOperation):
    def __init__(self):
        super().__init__()
        self.operation = "divide"

    def isSymmetric(self):
        return False

def getAllTwoArgOperations():
    return [SumOperation(), SubtractOperation(), MultiplyOperation(), DivideOperation()]
