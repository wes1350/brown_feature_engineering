from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd

class AggregateOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "aggregate"

    def is_redundant_to_self(self):
        return True

    def transform(self, df, new_feature_cache=None, concatenate_originals=True):
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()  # all numeric cols, including dummies
        categorical_dummy_cols = [col for col in df.columns.values.tolist() if "__dummy__" in col]  # just dummies
        if len(categorical_dummy_cols) != 0:  # Can happen if we feature select all the categorical variables out.
            # If this happens, an error will trigger later when adding the final dummy
            # variable result, so we just check to see if it's empty here.
            valid_numeric_cols = [col for col in numeric_col_names if
                                  col not in categorical_dummy_cols]  # numeric, not dummies

            # Far too many columns - need to merge (add) columns that correspond to the same categorical variable

            cat_var_to_agg_values = {}
            dummy_col_pos_values = {}
            for cCol in categorical_dummy_cols:
                dummy_col_pos_values[cCol] = np.flatnonzero(df[cCol] == 1)  # Find indices where true, so we can work
                # with numpy instead of pandas series overhead

            for nCol in valid_numeric_cols:
                n_col_loc = df.columns.get_loc(nCol)
                for cCol in categorical_dummy_cols:
                    cat_prefix = cCol.split("__dummy__")[0]
                    new_name = self.getOperation() + "(" + nCol + ", " + cat_prefix + ")"

                    if new_name not in set(df.columns):
                        if new_feature_cache is not None:
                            if new_name in new_feature_cache:
                                if new_name not in cat_var_to_agg_values:
                                    cat_var_to_agg_values[new_name] = new_feature_cache[new_name]
                                continue  # Don't recalculate if found in cache

                        if new_name not in cat_var_to_agg_values:
                            cat_var_to_agg_values[new_name] = np.array([0.0] * df.shape[0])

                        # Here we get all the values of the numeric column where the categorical variable equals 1
                        desired_numeric_vals = df.iloc[dummy_col_pos_values[cCol], n_col_loc]
                        cat_var_to_agg_values[new_name][dummy_col_pos_values[cCol]] = Tr.aggregateOperationWrapper(
                            self.getOperation(), desired_numeric_vals.values)

            new_values = np.zeros((df.shape[0], len(cat_var_to_agg_values)))  # will hold the new column values of df

            new_cols = list(cat_var_to_agg_values.keys())
            if new_feature_cache is None:
                for i in range(len(new_cols)):
                    new_values[:, i] = cat_var_to_agg_values[new_cols[i]]
            else:
                for i in range(len(new_cols)):
                    new_values[:, i] = cat_var_to_agg_values[new_cols[i]]
                    if new_cols[i] not in new_feature_cache:
                        new_feature_cache[new_cols[i]] = new_values[:, i]

            if concatenate_originals:
                return pd.concat([df, pd.DataFrame(new_values, columns=new_cols, index=df.index)], axis=1)
            else:
                return pd.DataFrame(new_values, columns=new_cols, index=df.index)
        else:
            return df

class MinOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "min_agg"

class MaxOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "max_agg"

class MeanOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "mean_agg"

class CountOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "count_agg"

class StdOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "std_agg"

class ZScoreOperation(AggregateOperation):
    def __init__(self):
        super().__init__()
        self.operation = "zscore_agg"

def getAllAggregateOperations():
    return [MinOperation(), MaxOperation(), MeanOperation(), CountOperation(), StdOperation(), ZScoreOperation()]
