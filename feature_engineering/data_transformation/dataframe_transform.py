import typing
import json
import os

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, featurization
from d3m import utils

from .operations import UnionOperation, FeatureSelectionOperation, SpatialAggregationOperation, OneArgOperation, \
    TwoArgOperation, AggregateOperation, CompactOneHotOperation, DateSplitOperation, FrequencyOperation, \
    StatisticalOperation, InitializationOperation
from . import FeatureSet
from . import Transformer

__all__ = ('DataframeTransformPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    paths = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='List of lists describing the tree structure leading to the desired result in json format.'
    )
    operations = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Dict of operation types corresponding to each of the labels given in paths in json format.'
    )
    names_to_keep = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Feature names to keep. None signifies keep all.'
    )
    opt_outs = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Dict of pre-processing steps to skip in json format.'
    )


class DataframeTransform(featurization.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which transforms a dataframe by adding or removing columns based on the operations specified. Additional
    columns take values based on previous columns, e.g. log of column values, or sum of two different column values.
    Columns can also be removed with certain operations. Preprocessing steps are also included, but can be opted out of.

    The primitive inputs represent a DAG originating from one node and leading to another. The DAG may "expand" into
    multiple paths by using several operations (described below) at non-terminal nodes, but ultimately must converge
    back into a single node. Each node must have an associated operation, which corresponds to some transformation
    function.

    The DAG structure is given by the paths variable below, while the operations which correspond to each node are
    specified in the operations hyperparameter.

    Example 1:

    We want to add to our dataframe the columns corresponding to the log of each numerical column.

    The DAG looks like this: 0 (base) -> 1 (log)

    Example 2:

    We want to add columns corresponding to log, and also columns corresponding to date splitting

                                                                               -> 1 (log)
    The DAG could look like this (nodes 1 and 2 are interchangeable): 0 (base)                   -> 3 (union)
                                                                               -> 2 (date_split)
    i.e., paths to the terminal node (3) include 0 -> 1 -> 3 and 0 -> 2 -> 3.
    Here, union just performs the union of the two dataframes generated (log result, and date_split result)

    Note that the following would produce the same result:

    0 (base) -> 1 (log) -> 2 (date_split)

    As log only acts on numeric columns and date_split only on datetime columns, the result of the log call is not
    influenced by the date_split operation. Hence union is actually unnecessary here.

    Example 3:

    We want to add columns corresponding to log of each numeric column, sqrt of each numeric column, and the sqrt of the
    log of each numeric column. Then, we only want to keep a specific subset of the generated columns.

    In this case, the DAG would look like this: 0 (base) -> 1 (sqrt) -> 2 (log)

    Let the set of original numeric columns be C. Then at node 1, we will have generated a dataframe with columns C and
    sqrt(C), where log C is a set of columns containing exactly all columns of C after a sqrt operation. In other words,
    the size of the numeric portion of the base dataframe has doubled in column dimension.

    After step 2, we will have generated the sqrt of each column in Union(C, sqrt C), as in step 1 we just added sqrt C
    to the original C dataframe. So, the dataframe will now be Union(C, sqrt C, log C, log sqrt C), and the dataframe
    will have quadrupled in size in column space.



    All hyperparameters described below are to be given in json format.


    paths: A list of lists describing paths from the base dataframe to the final result. All paths must start with the
    first node (0) and end with the final node, whatever it may be. All nodes specified in each path must have
    corresponding operations specified in the operations argument.
    -- Example: [[0, 1, 2, 5, 6], [0, 1, 3, 5, 6], [0, 4, 6]]
    -- Example: [[0, 1]] (simplest, will just do one transformation step)

    operations: A dict mapping nodes of the DAG to operation codes. The initial node operation is not specified (or can
                be specified with the dummy operation "INIT")
    -- Example: {1: "log", 2: "sum"} or {0: "INIT", 1: "log", 2: "sum"}

    names_to_keep: A list of strings corresponding to columns to retain after creation. Useful if we don't want to
    keep all generated columns. The default value of None retains all generated columns.

    opt_outs: A list of preprocessing steps to opt out of. Default is to opt out of none.

    -- Options: "skip_all": skip all preprocessing steps
                "skip_drop_index": drop the column "d3mIndex" if it exists
                "skip_infer_dates": don't try to produce datetime columns from names containing "date" or "Date"
                "skip_remove_full_NA_columns": don't remove columns whose values are all NA
                "skip_fill_in_categorical_NAs": don't replace NA values with the value "__missing__" in categorical cols
                "skip_impute_with_median": don't impute NA values with the median in numeric columns
                "skip_one_hot_encode": don't one hot encode categorical columns
                "skip_remove_high_cardinality_cat_vars": don't remove categorical columns with mostly unique values
                "skip_rename_for_xgb": don't rename columns to comply with XGBoost requirements
    """

    TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )
    REPOSITORY = 'https://gitlab.datadrivendiscovery.org/wrunnels/brown_feature_engineering'
    if TAG_NAME:
        PACKAGE_URI = "git+" + REPOSITORY + "@" + TAG_NAME
    else:
        PACKAGE_URI = "git+" + REPOSITORY
    PACKAGE_NAME = 'brown_feature_engineering'
    PACKAGE_URI = PACKAGE_URI + "#egg=" + PACKAGE_NAME

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '99951ce7-193a-408d-96f0-87164b9a2b26',
            'version': '0.1.0',
            'name': "Feature Engineering Dataframe Transformer",
            'keywords': ['feature engineering', 'transform'],
            'python_path': 'd3m.primitives.data_transformation.feature_transform.Brown',
            'source': {
                'name': "Brown",
                'contact': 'mailto:wrunnels@mit.edu',
                'uris': [
                    'https://gitlab.datadrivendiscovery.org/wrunnels/brown_feature_engineering',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                # 'package_uri': 'git+https://gitlab.datadrivendiscovery.org/wrunnels/brown_feature_engineering@master#egg=brown_feature_engineering'
                'package_uri': PACKAGE_URI
            }],
            'algorithm_types': ["DATA_CONVERSION"],
            'primitive_family': "FEATURE_EXTRACTION",
            'hyperparams_to_tune': ["paths", "operations", "names_to_keep", "opt_outs"]
        }
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """

        :param inputs: a pandas dataframe wrapped in a d3m container type
        :param timeout: ignored
        :param iterations: ignored
        :return: transformed dataframe in d3m type
        """
        # Translate json preprocessing opt outs
        if "opt_outs" not in self.hyperparams or self.hyperparams["opt_outs"] is None:
            translated_opt_outs = None
        else:
            translated_opt_outs = json.loads(self.hyperparams["opt_outs"])

        # Not enough info, just return original data
        if "paths" not in self.hyperparams or "operations" not in self.hyperparams:
            return inputs
        if self.hyperparams["paths"] is None or self.hyperparams["operations"] is None:
            return inputs

        # Translate rest of json inputs

        translated_paths = json.loads(self.hyperparams["paths"])
        translated_op_dict = json.loads(self.hyperparams["operations"])
        if "names_to_keep" not in self.hyperparams or self.hyperparams["names_to_keep"] is None:
            translated_names = None
        else:
            translated_names = json.loads(self.hyperparams["names_to_keep"])

        # reconstruct ops used to create for applyBulkTransform
        reconstructed_op_dict = {}
        for label in translated_op_dict:
            if translated_op_dict[label] == "INIT":
                reconstructed_op_dict[int(label)] = InitializationOperation.InitializationOperation()
            elif translated_op_dict[label] == "max":
                reconstructed_op_dict[int(label)] = AggregateOperation.MaxOperation()
            elif translated_op_dict[label] == "min":
                reconstructed_op_dict[int(label)] = AggregateOperation.MinOperation()
            elif translated_op_dict[label] == "mean":
                reconstructed_op_dict[int(label)] = AggregateOperation.MeanOperation()
            elif translated_op_dict[label] == "count":
                reconstructed_op_dict[int(label)] = AggregateOperation.CountOperation()
            elif translated_op_dict[label] == "std":
                reconstructed_op_dict[int(label)] = AggregateOperation.StdOperation()
            elif translated_op_dict[label] == "z_agg":
                reconstructed_op_dict[int(label)] = AggregateOperation.ZAggOperation()
            elif translated_op_dict[label] == "date_split":
                reconstructed_op_dict[int(label)] = DateSplitOperation.DateSplitOperation()
            elif translated_op_dict[label] == "feature_selection":
                reconstructed_op_dict[int(label)] = FeatureSelectionOperation.FeatureSelectionOperation()
            elif translated_op_dict[label] == "log":
                reconstructed_op_dict[int(label)] = OneArgOperation.LogOperation()
            elif translated_op_dict[label] == "sin":
                reconstructed_op_dict[int(label)] = OneArgOperation.SinOperation()
            elif translated_op_dict[label] == "cos":
                reconstructed_op_dict[int(label)] = OneArgOperation.CosOperation()
            elif translated_op_dict[label] == "tanh":
                reconstructed_op_dict[int(label)] = OneArgOperation.TanhOperation()
            elif translated_op_dict[label] == "rc":
                reconstructed_op_dict[int(label)] = OneArgOperation.ReciprocalOperation()
            elif translated_op_dict[label] == "square":
                reconstructed_op_dict[int(label)] = OneArgOperation.SquareOperation()
            elif translated_op_dict[label] == "sqrt":
                reconstructed_op_dict[int(label)] = OneArgOperation.SqrtOperation()
            elif translated_op_dict[label] == "sigmoid":
                reconstructed_op_dict[int(label)] = OneArgOperation.SigmoidOperation()
            elif translated_op_dict[label] == "sum":
                reconstructed_op_dict[int(label)] = TwoArgOperation.SumOperation()
            elif translated_op_dict[label] == "subtract":
                reconstructed_op_dict[int(label)] = TwoArgOperation.SubtractOperation()
            elif translated_op_dict[label] == "multiply":
                reconstructed_op_dict[int(label)] = TwoArgOperation.MultiplyOperation()
            elif translated_op_dict[label] == "divide":
                reconstructed_op_dict[int(label)] = TwoArgOperation.DivideOperation()
            elif translated_op_dict[label] == "zscore":
                reconstructed_op_dict[int(label)] = StatisticalOperation.ZScoreOperation()
            elif translated_op_dict[label] == "min_max_norm":
                reconstructed_op_dict[int(label)] = StatisticalOperation.MinMaxNormOperation()
            elif translated_op_dict[label] == "binning_u":
                reconstructed_op_dict[int(label)] = StatisticalOperation.BinningUOperation()
            elif translated_op_dict[label] == "binning_d":
                reconstructed_op_dict[int(label)] = StatisticalOperation.BinningDOperation()
            elif translated_op_dict[label] == "union":
                reconstructed_op_dict[int(label)] = UnionOperation.UnionOperation()
            elif translated_op_dict[label] == "spatial_min":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialMinOperation()
            elif translated_op_dict[label] == "spatial_max":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialMaxOperation()
            elif translated_op_dict[label] == "spatial_mean":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialMeanOperation()
            elif translated_op_dict[label] == "spatial_count":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialCountOperation()
            elif translated_op_dict[label] == "spatial_std":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialStdOperation()
            elif translated_op_dict[label] == "spatial_z_agg":
                reconstructed_op_dict[int(label)] = SpatialAggregationOperation.SpatialZAggOperation()
            elif translated_op_dict[label] == "one_term_frequency":
                reconstructed_op_dict[int(label)] = FrequencyOperation.OneTermFrequencyOperation()
            elif translated_op_dict[label] == "two_term_frequency":
                reconstructed_op_dict[int(label)] = FrequencyOperation.TwoTermFrequencyOperation()
            elif translated_op_dict[label] == "compact_one_hot":
                reconstructed_op_dict[int(label)] = CompactOneHotOperation.CompactOneHotOperation()
            else:
                raise ValueError("Could not identify an operation with given name: " + translated_op_dict[label])

        # Now check to make sure paths is valid, remove the final node from the path, and note which node to construct
        final_node_list = []
        for path in translated_paths:
            final_node_list.append(path.pop(-1))
        if len(final_node_list) > 1:
            for node in final_node_list[1:]:
                if node != final_node_list[0]:
                    raise ValueError("Some of the given paths terminate with different nodes")
        desired_node = final_node_list[0]

        # Make sure all paths begin from the same node
        if len(translated_paths) > 0:
            initial_node = translated_paths[0][0]
            for path in translated_paths:
                if path[0] != initial_node:
                    raise ValueError("Some paths have different starting nodes")

        # If initial node does not have the INIT operation mapped to it, add it automatically.
        # If it is already specified with the INIT operation, keep it
        if len(translated_paths) > 0:
            initial_node = translated_paths[0][0]
            if initial_node not in reconstructed_op_dict:
                reconstructed_op_dict[initial_node] = InitializationOperation.InitializationOperation()
            else:
                # If it is already specified with a different operation, raise Exception
                if translated_op_dict[str(initial_node)] != "INIT":
                    raise ValueError("Initial node specified with non-initialization operation. Nodes are specified "
                                     "with the operations that yield them from their parents, and the root node has no "
                                     "parent.")

        if desired_node not in reconstructed_op_dict:
            raise ValueError("Node to be constructed not found in given operation dictionary")
        for path in translated_paths:
            for node in path:
                if node not in reconstructed_op_dict:
                    raise ValueError("Cannot find node " + str(node) + " in given operation dictionary")
        # Initialize FeatureSet for preprocessing, and get its features

        root_features = FeatureSet.FeatureSet(data=inputs, only_reconstructing_new_data=True,
                                              preprocess_during_reconstruct=True,
                                              reconstruction_preprocessing_opt_outs=translated_opt_outs).getFeatures()

        # Now check to see if the current node has one or two parents
        first_parent = None
        second_parent = None
        for path in translated_paths:
            if first_parent is None:
                first_parent = path[-1]
            elif path[-1] != first_parent:
                if second_parent is not None:
                    if path[-1] != second_parent:
                        raise ValueError("Paths indicate desired node has more than 2 parents!")
                else:
                    second_parent = path[-1]

        if second_parent is None:
            # Make sure if our node is a union, we have at least 2 paths
            if translated_op_dict[str(desired_node)] == "union":
                raise ValueError("If specifying just one path, cannot specify Union operation for the desired node")
            reconstructed = Transformer.applyBulkTransform(root_features, reconstructed_op_dict[desired_node],
                                                           nodeOpDict=reconstructed_op_dict,
                                                           firstPathsFromRoot=translated_paths,
                                                           skip_fs_for_reconstruction=True)
        else:
            first_paths = []
            second_paths = []
            for path in translated_paths:
                if path[-1] == first_parent:
                    first_paths.append(path)
                else:
                    second_paths.append(path)

            reconstructed = Transformer.applyBulkTransform(root_features, reconstructed_op_dict[desired_node],
                                                           nodeOpDict=reconstructed_op_dict,
                                                           firstPathsFromRoot=first_paths,
                                                           secondPathsFromRoot=second_paths,
                                                           skip_fs_for_reconstruction=True)

        # If categorical variable values unseen in the previous run are now present, keep them
        full_cols = reconstructed.columns.tolist()
        dummy_cols = [name for name in full_cols if "__dummy__" in name]

        # Remove excess features since we didn't do feature selection in reconstruction
        # However, if the given set of names is None, assume we keep all given features
        if translated_names is not None:
            # Also we might not have some categorical variable values we had before in the new data. In this case,
            # to prevent errors we remove those from the list of translated names, as they don't exist in this new data

            names_to_remove = [name for name in translated_names if "__dummy__" in name and name not in dummy_cols]
            valid_names = [name for name in translated_names + dummy_cols if name not in names_to_remove]

            # Remove duplicate columns too; not sure if that will affect things yet
            try:
                reconstructed = reconstructed[list(set(valid_names + dummy_cols))]

                # Also remove any date time features we created
                reconstructed = reconstructed.select_dtypes(exclude="datetime")
            except:
                print(translated_names)
                print(reconstructed.columns)
                assert False

        # Remove dummy columns
        results = Transformer.recompress_categorical_features(reconstructed)

        outputs = container.DataFrame(results)

        outputs.metadata = inputs.metadata.generate(value=outputs)

        return base.CallResult(outputs)
