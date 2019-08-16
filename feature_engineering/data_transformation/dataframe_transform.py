import collections
import os
import typing
import json

from d3m import container#, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, featurization

# import common_primitives
# from common_primitives import dataframe_utils

from .operations import UnionOperation, FeatureSelectionOperation, SpatialAggregationOperation, OneArgOperation, \
    TwoArgOperation, AggregateOperation, CompactOneHotOperation, DateSplitOperation, FrequencyOperation, \
    StatisticalOperation, InitializationOperation
from . import FeatureSet
from . import Transformer

# __all__ = ('DataFrameTransformPrimitive',)

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


class DataframeTransformPrimitive(featurization.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which transforms a dataframe by adding or removing columns based on the operations specified. Additional
    columns take values based on previous columns, e.g. log of column values, or sum of two different column values.
    Columns can also be removed with certain operations. Preprocessing steps are also included, but can be opted out of.

    To use this, do [specify paths, ops, etc. conventions]
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '99951ce7-193a-408d-96f0-87164b9a2b26',
            'version': '0.1.0',
            'name': "Feature Engineering Dataframe Transformer",
            'keywords': ['feature engineering', 'transform'],
            'python_path': '#TODO',
            'source': {
                'name': "Brown",
                'contact': 'wrunnels@mit.edu',
                'uris': [
                    '#TODO - link to file',
                    '#TODO - link to repo'
                ],
            },
            'installation': [{
                'type': '#TODO',
                'package_uri': '#TODO'
            }],
            'algorithm_types': '#TODO',
            'primitive_family': '#TODO'
        }
    )

    def produce(self, *, inputs: Inputs,  paths : str = None, operations : str = None, names_to_keep : str = None,
                opt_outs : str = None) -> base.CallResult[Outputs]:
        # Translate json opt outs
        if opt_outs is not None:
            translated_opt_outs = json.loads(opt_outs)
        else:
            translated_opt_outs = None

        if paths is None or operations is None:
            return inputs  # Not enough info, just return original data

        # Translate rest of json inputs

        translated_paths = json.loads(paths)
        translated_op_dict = json.loads(operations)
        translated_names = json.loads(names_to_keep)

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
                    raise Exception("Some of the given paths terminate with different nodes")
        desired_node = final_node_list[0]

        # More checks? Like if all given labels in paths exist in op dict?
        if desired_node not in reconstructed_op_dict:
            print(reconstructed_op_dict, translated_op_dict, operations)
            print(desired_node)
            print(paths, translated_paths)
            print(names_to_keep, translated_names)
            raise Exception("Node to be constructed not found in given operation dictionary")
        for path in translated_paths:
            for node in path:
                if node not in reconstructed_op_dict:
                    raise Exception("Cannot find node " + str(node) + " in given operation dictionary")
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
                        raise Exception("Paths indicate desired node has more than 2 parents!")
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
        outputs = Transformer.recompress_categorical_features(reconstructed)

        return base.CallResult(outputs)

    # TODO: Update metadata ourselves? See IncrementPrimitive under TA1 Examples, for example
    # TODO: Specify target, inputs, etc. - through metadata? #6 on checklist