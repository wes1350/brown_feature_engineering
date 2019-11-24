from .operations.UnionOperation import UnionOperation
from .operations.DateSplitOperation import DateSplitOperation
from .operations.FeatureSelectionOperation import FeatureSelectionOperation
from .operations.CompactOneHotOperation import CompactOneHotOperation

import pandas as pd
from sklearn.preprocessing import Imputer


def applyBulkTransform(rootDF, op, nodeOpDict, firstPathsFromRoot, secondPathsFromRoot=None, target_for_fs=None,
                       splits_for_fs=None, target_type_for_fs=None, new_feature_cache=None, n_cores=1,
                       skip_fs_for_reconstruction=False, only_use_train_data=False, n_root_features=None,
                       fs_cap_factor=None, auto_fs=False, also_auto_fs_result=False, keep_originals_in_fs=False,
                       original_features=None):
    """
    Given a base dataframe and the operations necessary to construct a dataframe starting from the base dataframe,
    returns that dataframe. Chains calls of applyTransform together.

    :param rootDF: dataset corresponding to the root node
    :param op: The operation to use to create this node from its parent(s)
    :param nodeOpDict: a dict that for each node in pathsFromRoot says which operation was used to create it
    :param firstPathsFromRoot: a list of paths from the root to the first parent
    :param secondPathsFromRoot: a list of paths from the root to the second parent, in the case of unions
    :param target_for_fs: target values
    :param splits_for_fs: train/test split for use in feature selection
    :param target_type_for_fs: target type for use in feature selection
    :param new_feature_cache: cache of calculated features
    :param n_cores: cores to use during feature selection
    :param skip_fs_for_reconstruction: skip feature selection as we are reconstructing
    :param only_use_train_data: only look at train data (for feature selection)
    :return: A pandas dataframe as described above
    """

    # First, create a data structure that has (node, op, parent(s)) tuples, so we can identify how to create new nodes
    # Also keep track of how often each node appears in a path, for later usage when reconstructing datasets

    node_relations = {} # maps: node -> (operation used to create, parent 1, [optional] parent 2)
    node_appearance_counts = {} # Keeps track of how many times we access each node so we know when to remove references

    for path in firstPathsFromRoot if secondPathsFromRoot is None else firstPathsFromRoot + secondPathsFromRoot:
        for i in range(1, len(path)):
            if nodeOpDict[path[i]].isUnion():
                if path[i] in node_relations:
                    # only update if we are in intermediate state and have found the partner node in the union
                    if len(node_relations[path[i]]) == 2 and node_relations[path[i]][1] != path[i - 1]:
                        node_relations[path[i]] = (node_relations[path[i]][0], node_relations[path[i]][1], path[i - 1])
                else:
                    node_relations[path[i]] = (nodeOpDict[path[i]], path[i - 1])  # temporary state, will be updated
            else:
                node_relations[path[i]] = (nodeOpDict[path[i]], path[i - 1])
        for label in path:
            if label in node_appearance_counts:
                node_appearance_counts[label] += 1
            else:
                node_appearance_counts[label] = 1

    # Sorted list of descendants in the tree. We know if label1 < label2, node label2 is not an ancestor of node label1
    root_descendants = sorted(node_relations.keys())

    # Next, for each ancestor node, we calculate its dataset, clearing references to them as we finish

    calculated_datasets = {0: rootDF}  # keeps track of the produced data sets we create for each ancestor

    def fs_for_auto_fs(dataframe):
        categorical_dummy_cols = [col for col in dataframe.columns.values.tolist() if "__dummy__" in col]  # get dummies
        num_numeric_features = dataframe.select_dtypes(exclude="datetime").shape[1] - len(categorical_dummy_cols)

        # only feature select if # numeric features exploded; don't penalize for latent other types of features
        if fs_cap_factor is None or num_numeric_features > fs_cap_factor * n_root_features:
            return FeatureSelectionOperation.transform(df=dataframe, target_for_fs=target_for_fs,
                                                       splits_for_fs=splits_for_fs,
                                                       target_type_for_fs=target_type_for_fs, n_cores=n_cores,
                                                       skip_fs_for_reconstruction=skip_fs_for_reconstruction,
                                                       only_use_train_data=only_use_train_data,
                                                       n_root_features=n_root_features, fs_cap_factor=fs_cap_factor,
                                                       keep_originals=keep_originals_in_fs,
                                                       original_features=original_features)
        return dataframe

    for node in root_descendants:
        node_info = node_relations[node]
        if len(node_info) == 2:  # non-union
            calculated_datasets[node] = applyTransform(calculated_datasets[node_info[1]], node_info[0],
                                                       target_for_fs=target_for_fs,
                                                       target_type_for_fs=target_type_for_fs,
                                                       splits_for_fs=splits_for_fs, new_feature_cache=new_feature_cache,
                                                       n_cores=n_cores,
                                                       skip_fs_for_reconstruction=skip_fs_for_reconstruction,
                                                       only_use_train_data=only_use_train_data,
                                                       n_root_features=n_root_features, fs_cap_factor=fs_cap_factor,
                                                       keep_originals_in_fs=keep_originals_in_fs,
                                                       original_features=original_features)
            if auto_fs:
                calculated_datasets[node] = fs_for_auto_fs(calculated_datasets[node])

            node_appearance_counts[node_info[1]] -= 1
            if node_appearance_counts[node_info[1]] == 0:  # Don't need to use this anymore, so clear to save space
                calculated_datasets[node_info[1]] = None
        elif len(node_info) == 3:  # union
            calculated_datasets[node] = applyUnionTransform(calculated_datasets[node_info[1]],
                                                            calculated_datasets[node_info[2]])

            if auto_fs:
                calculated_datasets[node] = fs_for_auto_fs(calculated_datasets[node])

            node_appearance_counts[node_info[1]] -= 1
            node_appearance_counts[node_info[2]] -= 1
            if node_appearance_counts[node_info[1]] == 0:  # Don't need to use this anymore, so clear to save space
                calculated_datasets[node_info[1]] = None
            if node_appearance_counts[node_info[2]] == 0:  # Don't need to use this anymore, so clear to save space
                calculated_datasets[node_info[2]] = None
        else:
            raise Exception("SIZE ERROR IN SAVE SPACE CODE")  # never been reached, but for possible debugging

    # Now parent data set has been calculated, so we apply transform if it is not a union
    if secondPathsFromRoot is None:
        if op.isUnion():
            raise Exception("Wanted union transform, but only gave tree of one parent!")

        parent = firstPathsFromRoot[0][-1]

        result = applyTransform(calculated_datasets[parent], op,
                                target_for_fs=target_for_fs, target_type_for_fs=target_type_for_fs,
                                splits_for_fs=splits_for_fs, new_feature_cache=new_feature_cache, n_cores=n_cores,
                                skip_fs_for_reconstruction=skip_fs_for_reconstruction,
                                only_use_train_data=only_use_train_data, n_root_features=n_root_features,
                                fs_cap_factor=fs_cap_factor, keep_originals_in_fs=keep_originals_in_fs,
                                original_features=original_features)

        # DOUBLE DIP. We should only do this if we wouldn't otherwise do so i.e. when we don't create a FeatureSet after
        if auto_fs and also_auto_fs_result:  # Never reached?
            return fs_for_auto_fs(result)
        return result

    else:  # Instead we are reconstructing a union node, so apply union transform instead
        if not op.isUnion():
            raise Exception("Gave two parent trees, but didn't give union operation!")

        first_parent = firstPathsFromRoot[0][-1]
        second_parent = secondPathsFromRoot[0][-1]
        result = applyUnionTransform(calculated_datasets[first_parent], calculated_datasets[second_parent])

        if auto_fs and also_auto_fs_result:  # Never reached?
            return fs_for_auto_fs(result)
        return result


def applyTransform(df, op, target_for_fs=None, splits_for_fs=None, target_type_for_fs=None, new_feature_cache=None,
                   n_cores=1, skip_fs_for_reconstruction=False, only_use_train_data=False, n_root_features=None,
                   fs_cap_factor=None, keep_originals_in_fs=False, original_features=None):
    """
    Given a dataframe and operation, return a new dataframe where the given operation is applied to applicable columns

    :param df: dataframe to transform
    :param op: operation to use
    :param target_for_fs: target values
    :param splits_for_fs: train/test split for use in feature selection
    :param target_type_for_fs: target type for use in feature selection
    :param new_feature_cache: cache of calculated features
    :param n_cores: cores to use during feature selection
    :param skip_fs_for_reconstruction: skip feature selection as we are reconstructing
    :param only_use_train_data: only look at train data (for feature selection)
    :return: A pandas dataframe as described above
    """

    if op.opType in ["one_arg", "two_arg", "statistical", "aggregate", "spatial_aggregation", "frequency"]:
        return op.transform(df=df, new_feature_cache=new_feature_cache)
    elif op.opType == "date":
        return DateSplitOperation.transform(df=df)
    elif op.opType == "compact_one_hot":
        return CompactOneHotOperation.transform(df=df)
    elif op.opType == "feature_selection":
        return FeatureSelectionOperation.transform(df=df, target_for_fs=target_for_fs, splits_for_fs=splits_for_fs,
                                                   target_type_for_fs=target_type_for_fs, n_cores=n_cores,
                                                   skip_fs_for_reconstruction=skip_fs_for_reconstruction,
                                                   only_use_train_data=only_use_train_data,
                                                   n_root_features=n_root_features, fs_cap_factor=fs_cap_factor,
                                                   use_mean_threshold=True, keep_originals=keep_originals_in_fs,
                                                   original_features=original_features)


def applyUnionTransform(df1, df2):
    """
    Union two dataframes.
    :param df1: first dataframe
    :param df2: second dataframe
    :return: the union of df1 and df2
    """

    return UnionOperation.transform(df1, df2)


def recompress_categorical_features(df):
    """
    Reverse one-hot encoding done by the feature engineering algorithm. Only affects dummy columns affected
    by the naming convention used here, not general one-hot encoding passed in beforehand.
    :param df: dataframe to recompress
    :return: dataframe with dummy columns compressed into original categorical columns
    """
    dummy_cols = [c for c in df.columns.tolist() if "__dummy__" in c]
    dummy_col_prefixes = set() # Holds the prefixes of the dummy cols, i.e. the name of the categorical variable
    for c in dummy_cols:
        dummy_col_prefixes.add(c.split("__dummy__")[0])
    for cat_var_name in dummy_col_prefixes:
        original_cat_var_col = pd.Series([""]*df.shape[0], index=df.index)
        relevant_dummies = [x for x in dummy_cols if x.split("__dummy__")[0] == cat_var_name]
        for c in relevant_dummies:
            # Found a dummy corresponding to this variable, reassign the original value to the places it equals 1
            # original_cat_var_col[np.flatnonzero(df[c] == 1)] = c.split("__dummy__")[1]
            original_cat_var_col[df.index[df[c] == 1].tolist()] = c.split("__dummy__")[1]
        df.drop(relevant_dummies, inplace=True, axis=1)
        df[cat_var_name] = original_cat_var_col
    return df


def preprocess(data_input, opt_outs=None):
    """
    Pre-process the data according to given parameters
    :param data_input: original data (pandas df) before pre-processing
    :param opt_outs: list (or None), specifying which pre=processing steps to opt -out- of
    :return: pre-processed data
    """

    # Don't edit the original dataframe; copy it first
    data = data_input.copy()
    if opt_outs is None:
        opt_outs = []
    # "all" signifies to skip all preprocessing steps
    if "skip_all" in opt_outs:
        return data
    # Drop index as it is redundant
    if "skip_drop_index" not in opt_outs:
        data.drop("d3mIndex", axis=1, errors="ignore", inplace=True)
    # Infer dates from columns with the word "date" or "Date" in them
    if "skip_infer_dates" not in opt_outs:
        new_date_cols = {}  # Hold new inferred date columns, so we don't double-infer new columns as we add them
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

        num_instances = data.shape[0]

        unique_cat_var_counts = {}
        for c in data.columns:
            if "__dummy__" in c:
                if c.split("__dummy__")[0] not in unique_cat_var_counts:
                    unique_cat_var_counts[c.split("__dummy__")[0]] = 1
                else:
                    unique_cat_var_counts[c.split("__dummy__")[0]] += 1

        prefixes_to_remove = []
        for c_prefix in unique_cat_var_counts:
            if unique_cat_var_counts[c_prefix] >= unique_value_proportion_deletion_threshold * num_instances:
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
        # if USE_XGB:
        new_names = [name.replace("[", "__lb__").replace("]", "__rb__").replace("<", "__lt__") for name in
                     data.columns]
        data.columns = new_names
    return data
