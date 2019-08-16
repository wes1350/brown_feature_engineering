try:
    import Transformer
    from operations import UnionOperation as uo
    from operations.FeatureSelectionOperation import FeatureSelectionOperation
except ModuleNotFoundError:
    from . import Transformer
    from .operations import UnionOperation as uo
    from .operations.FeatureSelectionOperation import FeatureSelectionOperation

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import json
import os
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
# from sklearn.impute import SimpleImputer

import xgboost as xgb

USE_XGB = False


class FeatureSet:
    def __init__(self, dataDirectory=None, data=None, targetData=None, typeOfTarget=None, splits=None,
                 scoringMetric=None, only_reconstructing_new_data=False, preprocess_during_reconstruct=False,
                 reconstruction_preprocessing_opt_outs=None, config=None):
        """
        Initialize a FeatureSet either from file or from input data
        :param dataDirectory: if reading from file, the directory that contains the necessary info
        :param data: if inputting data, the dataframe
        :param targetData: if inputting data, the target values
        :param typeOfTarget: if inputting data, the target data type. "categorical" for categorical variables
        :param splits: if inputting data, the test/train splits
        :param scoringMetric: if inputting data, the scoring metric to use
        :param only_reconstructing_new_data: if we only need this FeatureSet for reconstruction purposes,
               this tells us to skip time-consuming steps like evaluating the performance of the features
        :param config: configuration parameters
        """
        if dataDirectory is None: # Not reading from file
            if data is not None: # Make sure we input data if not reading from file
                if config is not None and config.input_test_dataset is not None: # directly inputting data
                    self.features = self.preprocess(data, config.preprocessing_opt_outs)
                    self.target = config.input_test_target
                    self.targetType = config.input_test_target_type
                    self.splits = None
                    self.scoringMetric = config.input_test_scoring_metric

                else: # Not directly inputting data, so receiving data from a parent
                    if not only_reconstructing_new_data:
                        if targetData is None:
                            raise Exception("Cannot give data set without specifying target data!")
                        elif typeOfTarget is None:
                            raise Exception("Cannot give data without specifying target type!")

                    if preprocess_during_reconstruct:
                        self.features = self.preprocess(data, reconstruction_preprocessing_opt_outs)
                    else:
                        self.features = data
                    self.target = targetData
                    self.targetType = typeOfTarget
                    self.splits = splits
                    self.scoringMetric = scoringMetric
            elif targetData is not None:
                raise Exception("Cannot give target data without accompanying data set!")
            else:
                raise Exception("Must include data directory or initial data!")

            # Set some instance variables for future use
            self.numFeatures = self.features.shape[1]
            self.numInstances = self.features.shape[0]
            self.featureTypeCounts = self.calculateFeatureTypeCounts()

    def preprocess(self, data_input, opt_outs=None):
        """
        Pre-process the data according to given parameters
        :param data_input: original data (pandas df) before pre-processing
        :param opt_outs: list (or None), specifying which pre=processing steps to opt -out- of
        :return: pre-processed data
        """

        if opt_outs is None:
            opt_outs = []
        # "all" signifies to skip all preprocessing steps
        if "skip_all" in opt_outs:
            return data_input

        # Don't edit the original dataframe; copy it first
        data = data_input.copy()

        # Drop index as it is redundant
        if "skip_drop_index" not in opt_outs:
            data.drop("d3mIndex", axis=1, errors="ignore", inplace=True)
        # Infer dates from columns with the word "date" or "Date" in them
        if "skip_infer_dates" not in opt_outs:
            new_date_cols = {} # Hold new inferred date columns, so we don't double-infer new columns as we add them
            for c in data.columns:
                if "date_inferred_date_dayofweek_inferred_date" == c:
                    raise Exception
                if "_inferred_date" in c:
                    continue
                if "date" in c or "Date" in c:
                    if "__dummy__" not in c:
                        try:
                            # input[c + "_inferred_date"] = pd.to_datetime(data[c], infer_datetime_format=True,
                            #                                                             errors="raise")
                            new_date_cols[c + "_inferred_date"] = pd.to_datetime(data[c], infer_datetime_format=True,
                                                                                 errors="raise")
                            # print("Inferred that feature \"" + c + "\" is a datetime feature ")
                        except ValueError:  # Could not parse date correctly, could possibly not be a date
                                            # feature despite name (or NA values present...)
                            pass
            for c in new_date_cols:
                data[c] = new_date_cols[c]
        # Handle NA or missing values

        # First, get rid of any columns that are all NAs, as they are useless
        if "skip_remove_full_NA_columns" not in opt_outs:
            full_na_columns = []
            for i in range(data.shape[1]):
                if data.iloc[:, i].isnull().all():
                    full_na_columns.append(i)
            if len(full_na_columns) > 0:
                # print("The following features contained only NA values, so they are being removed: ",
                #       [data.columns[x] for x in full_na_columns])
                data.drop([data.columns[x] for x in full_na_columns], axis=1, inplace=True)

        # Next, fill categorical variable NAs with "__missing__"
        if "skip_fill_in_categorical_NAs" not in opt_outs:
            categorical_column_names = data.select_dtypes(include=['object']).columns.tolist()

            for c in categorical_column_names:
                data[c] = data[c].fillna("__missing__")

        # Next, fill in NA values in numeric features with median column values
        if "skip_impute_with_median" not in opt_outs:
            try:
                imp_median = Imputer(strategy='median')
                numeric_subset = data.select_dtypes(include="number")

                if numeric_subset.shape[1] > 0 and numeric_subset.isnull().values.any():
                    # print("Data has NA values in numeric data, so replacing these values with median value by feature")
                    data = pd.concat([pd.DataFrame(imp_median.fit_transform(numeric_subset),
                                                            columns=numeric_subset.columns,
                                                            index=numeric_subset.index),
                                               data.select_dtypes(exclude="number")], axis=1)
            except Exception:
                # For some reason, could not impute the missing values. Has not occurred, but perhaps a possibility.
                # So in this case we just remove NA columns.
                # print("Could not impute missing values for some reason! Removing columns with NAs instead!")

                # Remove features with NA values, as our evaluation cannot handle it. Should be numeric and datetime features only now
                # nCols = data.shape[1]
                input_col_drop = data.dropna(axis=1)
                # newNCols = input_col_drop.shape[1]

                # if newNCols < nCols:
                #     pColsDropped = (nCols - newNCols) / nCols
                    # print("Data set had NA values, so we removed " + str(nCols - newNCols) + " of " + str(nCols) +
                    #       " original features " + "(" + str(round(100 * pColsDropped, 2)) + "%)")
                data = input_col_drop

        # One-hot encode the data using pandas get_dummies
        if "skip_one_hot_encode" not in opt_outs:
            data = pd.get_dummies(data, prefix_sep="__dummy__")

        # Some categorical variables are not useful, because there are not enough of the same values. For example, if
        # the variable is a string of a city name, and each data point is a different city, this variable will be of
        # no use. So, to trim our dataset and prevent a sometimes extreme excess of features, we remove such variables.

        # For now, we just remove variables if they are "categorical", but each value is unique.
        if "skip_remove_high_cardinality_cat_vars" not in opt_outs:
            unique_value_proportion_deletion_threshold = 0.8

            self.numInstances = data.shape[0]

            unique_cat_var_counts = {}
            for c in data.columns:
                if "__dummy__" in c:
                    if c.split("__dummy__")[0] not in unique_cat_var_counts:
                        unique_cat_var_counts[c.split("__dummy__")[0]] = 1
                    else:
                        unique_cat_var_counts[c.split("__dummy__")[0]] += 1

            prefixes_to_remove = []
            for c_prefix in unique_cat_var_counts:
                if unique_cat_var_counts[ c_prefix] >= unique_value_proportion_deletion_threshold * self.numInstances:
                    # each value of this categorical variable is unique
                    prefixes_to_remove.append(c_prefix)
                    # print("Identified the feature \"" + c_prefix + "\" as being an almost uniquely valued categorical"
                    #                                                " variable (p = " + str(
                    #     round(unique_cat_var_counts[c_prefix] / self.numInstances, 2)) + "). Deleting!")
            # Now remove the features we identified as being useless
            columns_to_remove = []
            for c in data.columns:
                if "__dummy__" in c:
                    if c.split("__dummy__")[0] in prefixes_to_remove:
                        columns_to_remove.append(c)
            data = data.drop(columns_to_remove, axis=1)

        # Rename features for XGBoost - it does not accept "[", "]" or "<" in feature names
        if "skip_rename_for_xgb" not in opt_outs:
            if USE_XGB:
                new_names = [name.replace("[", "__lb__").replace("]", "__rb__").replace("<", "__lt__") for name in
                             data.columns]
                data.columns = new_names
        return data

    def getNumFeatures(self):
        """Return the number of features in this FeatureSet."""
        return self.numFeatures

    def getNumInstances(self):
        """Return the number of instances (data points) in the data."""
        return self.numInstances

    def calculateFeatureTypeCounts(self):
        """Calculate and return the feature types present in the dataset and their frequency."""

        types = {"numeric": 0, "date": 0, "categorical": 0}

        cat_var_names = set()

        for i in range(self.getNumFeatures()):
            current_feature = self.features.iloc[:, i]
            if self.dummiesArePresent("__dummy__", current_feature):
                # dummy string being present in name means this corresponds to a categorical feature
                # Check this first, as dummies actually have numeric dtype and don't want to count them as numeric
                cat_var_names.add(current_feature.name.split("__dummy__")[0])
            elif pd.api.types.is_numeric_dtype(current_feature):
                types["numeric"] += 1
            elif pd.api.types.is_datetime64_any_dtype(current_feature):
                types["date"] += 1
            types["categorical"] = len(cat_var_names)
        return types

    def dummiesArePresent(self, dummy_str, series):
        """Return if dummy variables are present in a series"""
        return dummy_str in series.name

    def getFeatureTypeCounts(self):
        """Return calculated feature type counts"""
        return self.featureTypeCounts

    def createChildFeatureSet(self, op, rootDF=None, nodeOpDict=None, firstPathsFromRoot=None, new_feature_cache=None,
                              config=None):
        """
        Return a new FeatureSet object formed by applying the specified operation to this FeatureSet.
        :param op: operation to apply to this FeatureSet
        :param rootDF: if doing bulk transform, root data frame to use
        :param nodeOpDict: if doing bulk transform, dict mapping nodes to operations used to create them
        :param firstPathsFromRoot: if doing bulk transform, paths in tree from root to parent
        :param new_feature_cache: cache for calculated features
        :param config: configuration parameters
        :return: The new FeatureSet calculated
        """

        if config.calculate_from_scratch:
            new_features = Transformer.applyBulkTransform(rootDF, op, nodeOpDict, firstPathsFromRoot,
                                                          target_for_fs=self.target, target_type_for_fs=self.targetType,
                                                          splits_for_fs=self.splits,
                                                          new_feature_cache=new_feature_cache, n_cores=config.n_cores,
                                                          only_use_train_data=config.only_use_train_data,
                                                          n_root_features=config.n_root_features,
                                                          fs_cap_factor=config.fs_cap_factor,
                                                          auto_fs=config.auto_fs, keep_originals_in_fs=config.auto_fs,
                                                          original_features=config.original_feature_names)
            # print("DONE WITH BULK TRANSFORM")
        else:
            new_features = Transformer.applyTransform(self.features, op, target_for_fs=self.target,
                                                      target_type_for_fs=self.targetType, splits_for_fs=self.splits,
                                                      new_feature_cache=new_feature_cache, n_cores=config.n_cores,
                                                      only_use_train_data=config.only_use_train_data,
                                                      n_root_features=config.n_root_features,
                                                      fs_cap_factor=config.fs_cap_factor,
                                                      keep_originals_in_fs=config.auto_fs,
                                                      original_features=config.original_feature_names)

        return FeatureSet(data=new_features, targetData=self.target, typeOfTarget=self.targetType, splits=self.splits,
                          scoringMetric=self.scoringMetric, config=config)


    def createUnionChildFeatureSet(self, other, rootDF=None, nodeOpDict=None, firstPathsFromRoot=None,
                                   secondPathsFromRoot=None, config=None):
        """
        Return a new FeatureSet object formed by applying a union operation to this FeatureSet and another.
        :param other: other FeatureSet to union with this one
        :param rootDF: if doing bulk transform, root data frame to use
        :param nodeOpDict: if doing bulk transform, dict mapping nodes to operations used to create them
        :param firstPathsFromRoot: if doing bulk transform, paths in tree from root to parent 1
        :param secondPathsFromRoot: if doing bulk transform, paths in tree from root to parent 2
        :param config: configuration parameters
        """

        if config.calculate_from_scratch:
            new_features = Transformer.applyBulkTransform(rootDF, uo.UnionOperation(), nodeOpDict,
                                                          firstPathsFromRoot, secondPathsFromRoot,
                                                          target_for_fs=self.target, target_type_for_fs=self.targetType,
                                                          splits_for_fs=self.splits, n_cores=config.n_cores,
                                                          only_use_train_data=config.only_use_train_data,
                                                          n_root_features=config.n_root_features,
                                                          fs_cap_factor=config.fs_cap_factor,
                                                          auto_fs=config.auto_fs, keep_originals_in_fs=config.auto_fs,
                                                          original_features=config.original_feature_names)
        else:
            new_features = Transformer.applyUnionTransform(self.features, other.features)

        return FeatureSet(data=new_features, targetData=self.target, typeOfTarget=self.targetType, splits=self.splits,
                          scoringMetric=self.scoringMetric, config=config)


    def clearData(self):
        """Remove reference to the calculated dataset to free memory."""
        self.features = None


    def getFeatures(self):
        """Return the features in this dataset"""
        return self.features


    def setFeatures(self, data):
        """Set this FeatureSet's features"""
        self.features = data


    def featuresIsNone(self):
        """Return whether this FeatureSet has no features (i.e. reference was removed)"""
        return self.features is None
