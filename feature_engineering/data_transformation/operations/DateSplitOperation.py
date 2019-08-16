from .Operation import Operation
import numpy as np
import pandas as pd

class DateSplitOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = "date_split"
        self.opType = "date"

    def is_redundant_to_self(self):
        return True

    @staticmethod
    def transform(df):
        date_col_names = df.select_dtypes(include=['datetime']).columns.tolist()  # all dates

        new_date_feature_name_pairs = []
        date_feature_suffixes = ["year", "month", "day", "hour", "minute", "second", "microsecond",
                                 "nanosecond", "dayofweek"]
        original_col_set = set(df.columns)

        # Find new date component columns that are not duplicates
        for i in range(len(date_col_names)):
            for j in range(len(date_feature_suffixes)):
                new_name_pair = (date_col_names[i], date_feature_suffixes[j])
                if new_name_pair[0] + "_" + new_name_pair[1] not in original_col_set:
                    new_date_feature_name_pairs.append(new_name_pair)

        new_values = np.zeros((df.shape[0], len(new_date_feature_name_pairs)))

        for i in range(len(new_date_feature_name_pairs)):
            current_pair = new_date_feature_name_pairs[i]
            if current_pair[1] == "year":
                current_component = df[current_pair[0]].dt.year
            elif current_pair[1] == "month":
                current_component = df[current_pair[0]].dt.month
            elif current_pair[1] == "day":
                current_component = df[current_pair[0]].dt.day
            elif current_pair[1] == "hour":
                current_component = df[current_pair[0]].dt.hour
            elif current_pair[1] == "minute":
                current_component = df[current_pair[0]].dt.minute
            elif current_pair[1] == "second":
                current_component = df[current_pair[0]].dt.second
            elif current_pair[1] == "microsecond":
                current_component = df[current_pair[0]].dt.microsecond
            elif current_pair[1] == "nanosecond":
                current_component = df[current_pair[0]].dt.nanosecond
            elif current_pair[1] == "dayofweek":
                current_component = df[current_pair[0]].dt.dayofweek
            else:
                raise Exception("Invalid date component: ", current_pair[1])

            new_values[:, i] = current_component

        return pd.concat([df, pd.DataFrame(new_values, columns=[p[0] + "_" + p[1] for p in new_date_feature_name_pairs],
                                           index=df.index)], axis=1)
