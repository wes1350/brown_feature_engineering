from .Operation import Operation
import numpy as np
from . import TransformOperations as Tr
import pandas as pd
import bisect

class SpatialAggregationOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = None
        self.opType = "spatial_aggregation"
        self.radius_proportion = 0.1

    def is_redundant_to_self(self):
        return True

    def transform(self, df, new_feature_cache=None):
        numeric_col_names = df.select_dtypes(include=['number']).columns.tolist()
        numeric_col_names = [name for name in numeric_col_names if "__dummy__" not in name]
        numeric_col_names_set = set(numeric_col_names)
        unique_name_pairs = []
        for i in range(len(numeric_col_names)):
            for j in list(range(i)) + list(range(i + 1, len(numeric_col_names))):
                old_name_1 = numeric_col_names[i]
                old_name_2 = numeric_col_names[j]
                new_name = self.getOperation() + "(" + old_name_1 + ", " + old_name_2 + ")"
                if not new_name in numeric_col_names_set:
                    unique_name_pairs.append((old_name_1, old_name_2))

        new_values = np.zeros((df.shape[0], len(unique_name_pairs)))  # will hold the new column values of df

        if new_feature_cache is None:
            for p in range(len(unique_name_pairs)):
                print(p, len(unique_name_pairs))
                key_col, agg_col = unique_name_pairs[p]
                min_v = np.min(df[key_col])
                max_v = np.max(df[key_col])
                radius = self.radius_proportion*(max_v-min_v)
                new_col = np.zeros(df.shape[0])

                sorted_key_col = df[key_col].sort_values()
                # print(sorted_key_col)
                # assert False
                # i = 0
                # for x in sorted_key_col:
                #     print(i, x)
                #     i += 1

                for i in range(len(new_col)):
                    center = df[key_col].iloc[i]
                    # print(bisect.bisect_left(sorted_key_col.values, center-2))
                    # print(bisect.bisect(sorted_key_col.values, center+2))
                    # print(center)
                    # assert False

                    # neighborhood = df[agg_col][abs(df[key_col] - center) <= window_length]
                    left_bound = bisect.bisect_left(sorted_key_col.values, center - radius)
                    right_bound = bisect.bisect(sorted_key_col.values, center + radius)
                    # print(left_bound, right_bound)
                    # print("\n\n----\n\n")
                    # print(sorted_key_col.iloc[left_bound:right_bound].index)
                    # assert False
                    neighborhood = df[agg_col][sorted_key_col.iloc[left_bound:right_bound].index]
                    new_col[i] = Tr.aggregateOperationWrapper(self.getOperation(), neighborhood)
                    # print(new_col[i])

                new_values[:, p] = new_col
        else:
            for p in range(len(unique_name_pairs)):
                name = self.getOperation() + "(" + p[0] + ", " + p[1] + ")"
                if name in new_feature_cache:
                    new_values[:, p] = new_feature_cache[name]
                else:
                    key_col, agg_col = unique_name_pairs[p]
                    min_v = np.min(key_col)
                    max_v = np.max(key_col)
                    window_length = self.radius_proportion * (max_v - min_v)
                    new_col = np.array(df.shape[0])

                    for i in range(len(new_col)):
                        center = df[key_col].iloc[i]
                        neighborhood = df[agg_col][abs(df[key_col] - center) <= window_length]
                        new_col[i] = Tr.aggregateOperationWrapper(self.getOperation(), neighborhood)

                    new_values[:, p] = new_col
                    new_feature_cache[name] = new_col

        return pd.concat([df, pd.DataFrame(new_values, columns=[self.getOperation() + "(" + p[0] + ", " + p[1] + ")" for
                                                                p in unique_name_pairs], index=df.index)], axis=1)


class SpatialMinOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_min"

class SpatialMaxOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_max"

class SpatialMeanOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_mean"

class SpatialCountOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_count"

class SpatialStdOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_std"

class SpatialZAggOperation(SpatialAggregationOperation):
    def __init__(self):
        super().__init__()
        self.operation = "spatial_z_agg"

def getAllSpatialAggregateOperations():
    return [SpatialMinOperation(), SpatialMaxOperation(), SpatialMeanOperation(),
            SpatialCountOperation(), SpatialStdOperation(), SpatialZAggOperation()]
