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
    def transform(df, specified_cols=None, concatenate_originals=True):
        """
        Take a dataframe with datetime columns and add numeric features that extract info from them
        :param df: dataframe to transform
        :param specified_cols: list of certain features to create, e.g. ["year", "month", "day"]
                               if None, create all choices by default
        :return: transformed dataframe
        """
        date_col_names = df.select_dtypes(include=['datetime']).columns.tolist()  # all dates

        new_date_feature_name_pairs = []
        if specified_cols is None:
            date_feature_suffixes = ["year", "month", "day", "hour", "minute", "second", "microsecond",
                                     "nanosecond", "dayofweek"]
        else:
            if isinstance(specified_cols, str):
                date_feature_suffixes = [specified_cols]
            else:
                date_feature_suffixes = specified_cols

        original_col_set = set(df.columns)

        # Find new date component columns that are not duplicates
        for i in range(len(date_col_names)):
            for j in range(len(date_feature_suffixes)):
                new_name_pair = (date_col_names[i], date_feature_suffixes[j])
                new_name = "date_split_" + new_name_pair[1] + "(" + new_name_pair[0] + ")"
                if new_name not in original_col_set:
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

        if concatenate_originals:
            return pd.concat([df, pd.DataFrame(new_values, columns=["date_split_" + p[1] + "(" + p[0] + ")" for
                                                   p in new_date_feature_name_pairs], index=df.index)], axis=1)
        else:
            return pd.DataFrame(new_values, columns=["date_split_" + p[1] + "(" + p[0] + ")" for
                                                   p in new_date_feature_name_pairs], index=df.index)