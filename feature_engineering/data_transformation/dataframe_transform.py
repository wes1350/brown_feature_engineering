import typing
import json
import os

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, featurization
from d3m import utils

from .operations import OneArgOperation, TwoArgOperation, AggregateOperation, DateSplitOperation, FrequencyOperation,\
    StatisticalOperation, InitializationOperation
import pandas as pd

__all__ = ('DataframeTransformPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    features = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Feature names to generate.'
    )


class DataframeTransform(featurization.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which transforms a dataframe by adding or removing columns based on the columns initally present in the
    dataframe, e.g. log of column values, or sum of two different column values.


    features: A list of strings or json array corresponding to columns new columns to generate. Note that all
    original columns will be included in the returned dataframe, even if they are not included here.

    Returns a dataframe containing the columns in the original dataframe, plus new columns for each feature name given.
    """

    _TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )
    _REPOSITORY = 'https://github.com/wes1350/brown_feature_engineering'
    if _TAG_NAME:
        _PACKAGE_URI = "git+" + _REPOSITORY + "@" + _TAG_NAME
    else:
        _PACKAGE_URI = "git+" + _REPOSITORY
    _PACKAGE_NAME = 'brown_feature_engineering'
    _PACKAGE_URI = _PACKAGE_URI + "#egg=" + _PACKAGE_NAME

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '99951ce7-193a-408d-96f0-87164b9a2b26',
            'version': '0.2.1',
            'name': "Feature Engineering Dataframe Transformer",
            'keywords': ['feature engineering', 'transform'],
            'python_path': 'd3m.primitives.data_transformation.feature_transform.Brown',
            'source': {
                'name': "Brown",
                'contact': 'mailto:wrunnels@mit.edu',
                'uris': [
                    _REPOSITORY,
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                # 'package_uri': 'git+https://gitlab.datadrivendiscovery.org/wrunnels/brown_feature_engineering@master#egg=brown_feature_engineering'
                'package_uri': _PACKAGE_URI
            }],
            'algorithm_types': ["DATA_CONVERSION"],
            'primitive_family': "FEATURE_EXTRACTION",
            'hyperparams_to_tune': ["features"]
        }
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        :param inputs: a pandas dataframe wrapped in a d3m container type
        :param timeout: ignored
        :param iterations: ignored
        :return: transformed dataframe in d3m type
        """

        df = inputs

        op_mapping = {"INIT": InitializationOperation.InitializationOperation(),
                      "max_agg": AggregateOperation.MaxOperation(),
                      "min_agg": AggregateOperation.MinOperation(),
                      "mean_agg": AggregateOperation.MeanOperation(),
                      "count_agg": AggregateOperation.CountOperation(),
                      "std_agg": AggregateOperation.StdOperation(),
                      "zscore_agg": AggregateOperation.ZScoreOperation(),
                      "date_split": DateSplitOperation.DateSplitOperation(),
                      "log": OneArgOperation.LogOperation(),
                      "sin": OneArgOperation.SinOperation(),
                      "cos": OneArgOperation.CosOperation(),
                      "tanh": OneArgOperation.TanhOperation(),
                      "rc": OneArgOperation.ReciprocalOperation(),
                      "square": OneArgOperation.SquareOperation(),
                      "sqrt": OneArgOperation.SqrtOperation(),
                      "sigmoid": OneArgOperation.SigmoidOperation(),
                      "sum": TwoArgOperation.SumOperation(),
                      "subtract": TwoArgOperation.SubtractOperation(),
                      "multiply": TwoArgOperation.MultiplyOperation(),
                      "divide": TwoArgOperation.DivideOperation(),
                      "zscore": StatisticalOperation.ZScoreOperation(),
                      "min_max_norm": StatisticalOperation.MinMaxNormOperation(),
                      "binning_u": StatisticalOperation.BinningUOperation(),
                      "binning_d": StatisticalOperation.BinningDOperation(),
                      "one_term_frequency": FrequencyOperation.OneTermFrequencyOperation()}

        def generate_feature_from_name(feature, new_index):
            """Given a name to engineer, return a new feature corresponding to that name."""

            def parse_feature_name(feature_name):

                def takes_two_args(op_str):
                    """Return True if the operation corresponding to the input takes two arguments."""
                    return op_mapping[op_str].isAgg() or op_mapping[op_str].isTwoArg()

                def get_two_features_from_str(s):
                    """
                    For a given input string supposed to contain the names of two features, return them.
                    We do this by counting where parentheses balance in the input string.

                    e.g. "log(log(a))" -> "log(a)" only balances at end, so it only contains one feature
                         "sum(a, log(b))" -> "a, log(b)" has a ", " first, so it splits into two features
                         "sum(log(a), sin(b))" -> "log(a), sin(b)" balances out after log, so log(a) is one of the features
                         "sum(sum(log(a), sin(b)))" -> "sum(log(a), sin(b))" balances at the end, so it only contains one
                         "sum(cos(log(a)), sin(b))" -> "cos(log(a)), sin(b)" balances after cos(log), so it contains two
                    """
                    parenthesis_ct = 0
                    for i in range(len(s)):
                        if parenthesis_ct < 0:
                            raise Exception("Could not parse name due to invalid parenthesis structure!")
                        if parenthesis_ct == 0:
                            # Check to make sure at least 3 characters remain and the next 2 are ", "
                            if i < len(s) - 2 and s[i:i + 2] == ", ":
                                return s[:i], s[i + 2:]
                        if s[i] == "(":
                            parenthesis_ct += 1
                        elif s[i] == ")":
                            parenthesis_ct -= 1
                    # If we never returned, there must be an error in the feature name
                    raise ValueError("Could not parse name for two features: " + s)

                # If no parentheses, then this is not an engineered feature
                if "(" not in feature_name or ")" not in feature_name:
                    return {"op_name": "INIT", "feature_names": [feature_name], "additional_info": None}
                else:
                    for op_name in op_mapping:
                        # Check to see if the feature name begins with the op name
                        if feature_name[:len(op_name)] == op_name:
                            opless_name = feature_name[len(op_name):]
                            if op_name == "date_split":
                                if "(" not in opless_name or opless_name[0] != "_":
                                    raise Exception("Could not parse date split feature name: " + feature_name)
                                date_feature_type = opless_name.split("(")[0][
                                                    1:]  # try to extract "year", "day", etc.
                                # e.g. return ("date_split", "year", "feature_a") from "date_split_year(feature_a)"
                                return {"op_name": op_name,
                                        "feature_names": [opless_name[1 + len(date_feature_type) + 1:-1]],
                                        "additional_info": date_feature_type}
                            else:
                                # Make sure that parentheses enclose the other arguments
                                if len(opless_name) > 2 and opless_name[0] == "(" and opless_name[-1] == ")":
                                    trimmed_str = opless_name[1:-1]  # Get rid of parentheses too
                                    # If it takes two arguments, parse them
                                    if takes_two_args(op_name):
                                        arguments = get_two_features_from_str(trimmed_str)
                                        if arguments[0] == arguments[1]:
                                            raise ValueError("Cannot duplicate argument for operation taking two arguments:"
                                                             " {name}".format(name=feature_name))
                                        return {"op_name": op_name, "feature_names": [arguments[0], arguments[1]],
                                                "additional_info": None}
                                    else:  # If it only takes one argument, the trimmed name will be the feature name
                                        return {"op_name": op_name, "feature_names": [trimmed_str],
                                                "additional_info": None}
                                else:
                                    raise ValueError(
                                        "Could not parse name due to lack of parentheses or invalid feature name: " + feature_name)
                # If we reach here, did not find an adequate condition for parsing, so assume input is an original feature.
                return {"op_name": "INIT", "feature_names": [feature_name], "additional_info": None}

            feature_info_dict = parse_feature_name(feature)
            op_name = feature_info_dict["op_name"]
            args = feature_info_dict["feature_names"]
            if op_name == "INIT":
                return None
            elif op_mapping[op_name].opType in ["one_arg", "statistical", "frequency"]:
                values = df[args[0]].values if args[0] in df.columns else \
                generate_feature_from_name(args[0], new_index=new_index)[args[0]]
                return op_mapping[op_name].transform(pd.DataFrame({args[0]: values}, index=new_index),
                                                     concatenate_originals=False)
            elif op_mapping[op_name].opType == "aggregate":
                values_0 = df[args[0]].values if args[0] in df.columns else \
                generate_feature_from_name(args[0], new_index=new_index)[args[0]]
                values_1 = df[args[1]].values if args[1] in df.columns else \
                generate_feature_from_name(args[1], new_index=new_index)[args[1]]
                one_hot_df = pd.get_dummies(pd.DataFrame({args[0]: values_0, args[1]: values_1}, index=new_index),
                                            prefix_sep="__dummy__")
                return op_mapping[op_name].transform(one_hot_df, concatenate_originals=False)
            elif op_mapping[op_name].opType == "two_arg":
                values_0 = df[args[0]].values if args[0] in df.columns else \
                generate_feature_from_name(args[0], new_index=new_index)[args[0]]
                values_1 = df[args[1]].values if args[1] in df.columns else \
                generate_feature_from_name(args[1], new_index=new_index)[args[1]]
                return op_mapping[op_name].transform(
                    pd.DataFrame({args[0]: values_0, args[1]: values_1}, index=new_index),
                    correlation_threshold=2,  # threshold must be > 1 to ensure no removal
                    concatenate_originals=False)
            elif op_mapping[op_name].opType == "date":
                values = df[args[0]].values if args[0] in df.columns else \
                generate_feature_from_name(args[0], new_index=new_index)[args[0]]
                return op_mapping[op_name].transform(pd.DataFrame({args[0]: pd.to_datetime(values)}, index=new_index),
                                                     specified_cols=feature_info_dict["additional_info"],
                                                     concatenate_originals=False)
            else:
                raise Exception("Invalid operation: " + op_name)

        df_copy = df.copy()
        names = self.hyperparams["features"]
        if type(names) == "str":
            names = json.loads(names)

        for name in names:
            if name in df_copy.columns:
                continue
            generated_feature = generate_feature_from_name(name, new_index=df.index)
            if generated_feature is None:
                pass
            else:
                df_copy[name] = generated_feature[name]

        outputs = container.DataFrame(df.copy)

        outputs.metadata = inputs.metadata.generate(value=outputs)

        return base.CallResult(outputs)
