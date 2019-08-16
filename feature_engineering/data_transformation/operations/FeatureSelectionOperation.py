from .Operation import Operation
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

class FeatureSelectionOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = "feature_selection"
        self.opType = "feature_selection"

    @staticmethod
    def transform(df, target_for_fs=None, splits_for_fs=None, target_type_for_fs=None, n_cores=1,
                  skip_fs_for_reconstruction=False, only_use_train_data=False, n_root_features=None,
                  fs_cap_factor=None, use_mean_threshold=False, keep_originals=False, original_features=None):
        if skip_fs_for_reconstruction:
            return df

        if only_use_train_data:  # Note: only using train data doesn't reduce the size of data set, it just means for
                                #       feature selection purposes we only consider train data to choose features
            # print("ONLY USING TRAIN DATA!")

            # Use train_test_split to randomly shuffle the data - otherwise performance can be much worse
            # data, _, labels, _ = train_test_split(df.loc[splits_for_fs == "TRAIN"],
            #                                       target_for_fs.loc[splits_for_fs == "TRAIN"], test_size=0,
            #                                       random_state=1)
            data = df.loc[splits_for_fs == "TRAIN"].sample(frac=1, random_state=1)
            labels = target_for_fs.loc[splits_for_fs == "TRAIN"].sample(frac=1, random_state=1)
        else:
            # data, _, labels, _ = train_test_split(df, target_for_fs, test_size=0, random_state=1)
            data = df.sample(frac=1, random_state=1)
            labels = target_for_fs.sample(frac=1, random_state=1)

        categorical_dummy_cols = [col for col in df.columns.values.tolist() if "__dummy__" in col]  # just dummies

        if len(categorical_dummy_cols) == len(df.columns):
            print("Cannot do feature selection when only categorical features exist! Returning original dataframe.")
            return df

        numeric_data = data.select_dtypes(exclude="datetime").drop(columns=categorical_dummy_cols)

        # For auto-fs, we don't want to be too aggressive and constantly look to eliminate the original features.
        # Store the names
        original_names_to_keep = []
        if keep_originals:
            if original_features is None:
                raise ValueError("If trying to keep original features, must specify their names!")
            numeric_col_names = set(numeric_data.columns)
            original_names_to_keep += [name for name in original_features if name in numeric_col_names]
            numeric_data = numeric_data.drop(original_names_to_keep, axis=1)

            # Trying to do feature selection when only the original features remained; ignore and move on
            if numeric_data.shape[1] == 0:
                print("Tried to do feature selection when only original features remained, and was indicated to keep"
                      "original features, so no feature selection was done.")
                return df

        # Reduce bloat by setting an upper limit on the number of selected features as a multiple of the original number
        max_features = None
        if n_root_features is not None and fs_cap_factor is not None:
            # Take min because SelectFromModel throws error if max_features is higher than the given number of features
            # Subtract from the cap the number of original names we are keeping, so that new + original sums to cap
            max_features = min(fs_cap_factor * n_root_features - len(original_names_to_keep), numeric_data.shape[1])
            # print("MAX FEATURES:", max_features)
        threshold = "mean" if max_features is None or use_mean_threshold else -np.inf

        if target_type_for_fs == "categorical":
            sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=n_cores),
                                  max_features=max_features, threshold=threshold)
        else:
            sel = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=n_cores),
                                  max_features=max_features, threshold=threshold)

        # Fit the model, excluding datetime dtype columns, as we can't fit a model with them in the dataframe
        # Also exclude columns of categorical variables - don't want to feature select only some of the categories
        try:  # Generally, this will work just fine
            sel.fit(numeric_data, labels)
        except ValueError:  # If try failed, likely some values over capacity of float32, which random forest only uses
            # So, we just force values outside the float32 range to be the min/max values of float32
            # Might also throw a warning about overflow in reduce, but likely not too much of an issue
            s = numeric_data
            s = s.where(s < np.finfo(np.float32).max, np.finfo(np.float32).max)
            s = s.where(s > np.finfo(np.float32).min, np.finfo(np.float32).min)
            sel.fit(s, labels)

        selected_feat = data.select_dtypes(exclude="datetime") \
            .drop(columns=categorical_dummy_cols + original_names_to_keep).columns[sel.get_support()]

        # Add datetime and categorical dummy columns back in and return other selected columns
        # Also add original columns that we want to keep if we dropped them earlier

        if keep_originals and len(original_names_to_keep) > 0:
            return pd.concat([df.loc[:, original_names_to_keep], df.loc[:, selected_feat],
                              df.select_dtypes(include="datetime"), df.loc[:, categorical_dummy_cols]], axis=1)
        else:
            return pd.concat([df.loc[:, selected_feat], df.select_dtypes(include="datetime"),
                              df.loc[:, categorical_dummy_cols]], axis=1)
