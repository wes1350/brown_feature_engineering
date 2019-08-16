import pandas as pd
from .Config import Config
from .Graph import Graph
import warnings; warnings.filterwarnings("ignore")
from .operations import OneArgOperation, TwoArgOperation, StatisticalOperation, AggregateOperation, \
    DateSplitOperation, UnionOperation
from .Transformer import applyBulkTransform, applyTransform, applyUnionTransform

n1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n2 = pd.Series([3, 6, 10, 0, 0, 5, 2, -3, -7, 1])
n3 = pd.Series([77, 54, 124, -665, 104, 7, 16.3, 8.22, 19.102, 5])
c1 = pd.Series(['a', 'b', 'c', 'c', 'c', 'b', 'a', 'b', 'b', 'b'])
c2 = pd.Series(['X', 'Y', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'X', 'Y'])
date = pd.Series(['06/22/18', '05/22/18', '03/13/17', '12/01/19', '09/03/07', '09/04/19', '03/22/18', '05/22/18', '03/03/17', '03/03/16'])
# date = pd.to_datetime(date, infer_datetime_format=True, errors="raise")

d = {'n1': n1, 'n2': n2, 'n3': n3, 'c1': c1, 'c2': c2, 'date': date}
df = pd.DataFrame(data=d)
# print(df)
# print(df.dtypes)

t = pd.Series(['A', 'B', 'B', 'A', 'B', 'A', 'B', 'B', 'B', 'A'])

config = Config()
config.input_test_dataset = df
config.input_test_target = t
config.input_test_target_type = 'categorical'
config.preprocessing_opt_outs = []
config.max_height = 5

g = Graph(config=config)
# print(g.getNodeAtIndex(0).getFeatureSet().getFeatures().columns)
assert g.getFileName() == "Input" # Make sure file name is read properly
assert g.getNumNodes() == 1 # Make sure length works here
try: # See if an illegal addition is allowed
    g.addNode(1, OneArgOperation.SquareOperation(), config=config)
    raise Exception("Should have raised exception!")
except ValueError:
    pass

# Disable pre-processing for direct input after the first iteration, as it is redundant
config.preprocessing_opt_outs = ["skip_all"]

# Test Graph and classes that depend on it (Operation, Node, FeatureSet)

g.addNode(0, OneArgOperation.LogOperation(), config=config)
g.addNode(0, TwoArgOperation.SumOperation(), config=config)
g.addNode(0, DateSplitOperation.DateSplitOperation(), config=config)
g.addUnionNode(1, 3, UnionOperation.UnionOperation(), config=config)
g.addNode(2, StatisticalOperation.MinMaxNormOperation(), config=config)
g.addNode(2, AggregateOperation.MaxOperation(), config=config)
g.addNode(4, OneArgOperation.LogOperation(), config=config)
g.addUnionNode(6, 7, UnionOperation.UnionOperation(), config=config)
try: # See if an illegal addition is allowed
    g.addNode(8, OneArgOperation.SquareOperation(), config=config)
    raise Exception("Should have raised exception!")
except ValueError:
    pass

assert g.getNumNodes() == 9
assert g.getNodeAtIndex(0).isRootNode()
assert not g.getNodeAtIndex(1).isRootNode()
assert g.getParentLabels(2) == [0]
assert g.getChildrenLabels(2) == [5, 6]
assert g.getParentLabels(0) == []
assert g.getChildrenLabels(0) == [1, 2, 3]
assert isinstance(g.getNodeAtIndex(3).getOpUsedToCreate(), DateSplitOperation.DateSplitOperation)
assert g.getNodeDepth(0) == 1
assert g.getNodeDepth(1) == 2
assert g.getNodeDepth(5) == 3
assert g.getNodeDepth(7) == 4
paths7 = g.getPathsFromRootToNode(7)
assert [0, 1, 4, 7] in paths7
assert [0, 3, 4, 7] in paths7
assert len(paths7) == 2
paths8 = g.getPathsFromRootToNode(8)
assert [0, 3, 4, 7, 8] in paths8
assert [0, 1, 4, 7, 8] in paths8
assert [0, 2, 6, 8] in paths8
assert len(paths8) == 3
assert g.getNumberOfTimesOpUsedInPath([0, 1, 4, 7, 8], "union") == 2
assert g.getNumberOfTimesOpUsedInPath([0, 1, 4, 7, 8], "log") == 2
assert g.getNumberOfTimesOpUsedInPath([0, 3, 4, 7, 8], "date_split") == 1
assert g.getNumberOfTimesOpUsedInPath([0, 3, 4, 7, 8], "square") == 0
assert g.checkIfOpHasBeenUsedAtNode(0, OneArgOperation.LogOperation())
assert not g.checkIfOpHasBeenUsedAtNode(0, OneArgOperation.SquareOperation())
assert g.checkIfOpHasBeenUsedAtNode(3, UnionOperation.UnionOperation())
assert not g.checkIfOpHasBeenUsedAtNode(3, OneArgOperation.LogOperation())
assert g.checkIfTwoNodesHaveBeenUnioned(1, 3)
assert g.checkIfTwoNodesHaveBeenUnioned(6, 7)
assert not g.checkIfTwoNodesHaveBeenUnioned(1, 6)
assert len(g.getChildrenOperations(6)) == 1
assert g.getChildrenOperations(6)[0].getOperation() == "union"
assert len(g.getChildrenOperations(0)) == 3
ops = [op.getOperation() for op in g.getChildrenOperations(0)]
assert "union" not in ops
assert "date_split" in ops
assert "log" in ops
assert "sum" in ops
assert g.checkIfNodeIsAncestor(7, 1)
assert not g.checkIfNodeIsAncestor(1, 7)
assert not g.checkIfNodeIsAncestor(1, 6)
assert not g.checkIfNodeIsAncestor(6, 1)

# Test transform operations within each operation and Transformer

df2 = g.getNodeAtIndex(0).getFeatureSet().getFeatures()

def check_size(data, a, b):
    return data.shape[0] == a and data.shape[1] == b
assert check_size(df2, 10, 9)

def series_equals(s, l):
    s_list = s.tolist()
    if len(s_list) != len(l):
        return False
    for i in range(len(l)):
        if s_list[i] != l[i]:
            return False
    return True

def series_approx_equals(s, l):
    s_list = s.tolist()
    if len(s_list) != len(l):
        return False
    for i in range(len(l)):
        if s_list[i] >= l[i] + 0.001 or s_list[i] <= l[i] - 0.001:
            return False
    return True

assert series_equals(df2["n1"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

log = applyTransform(df2, OneArgOperation.LogOperation())
sin = applyTransform(df2, OneArgOperation.SinOperation())
cos = applyTransform(df2, OneArgOperation.CosOperation())
square = applyTransform(df2, OneArgOperation.SquareOperation())
sqrt = applyTransform(df2, OneArgOperation.SqrtOperation())
tanh = applyTransform(df2, OneArgOperation.TanhOperation())
sigmoid = applyTransform(df2, OneArgOperation.SigmoidOperation())
reciprocal = applyTransform(df2, OneArgOperation.ReciprocalOperation())
sum_ = applyTransform(df2, TwoArgOperation.SumOperation())
subtract = applyTransform(df2, TwoArgOperation.SubtractOperation())
multiply = applyTransform(df2, TwoArgOperation.MultiplyOperation())
divide = applyTransform(df2, TwoArgOperation.DivideOperation())
minmaxnorm = applyTransform(df2, StatisticalOperation.MinMaxNormOperation())
zscore = applyTransform(df2, StatisticalOperation.ZScoreOperation())
max_ = applyTransform(df2, AggregateOperation.MaxOperation())
min_ = applyTransform(df2, AggregateOperation.MinOperation())
std = applyTransform(df2, AggregateOperation.StdOperation())
count = applyTransform(df2, AggregateOperation.CountOperation())
mean = applyTransform(df2, AggregateOperation.MeanOperation())
z_agg = applyTransform(df2, AggregateOperation.ZAggOperation())
datesplit = applyTransform(df2, DateSplitOperation.DateSplitOperation())
df3 = g.getNodeAtIndex(1).getFeatureSet().getFeatures()
df4 = g.getNodeAtIndex(2).getFeatureSet().getFeatures()
union = applyUnionTransform(df3, df4)

assert check_size(log, 10, 12)
assert check_size(sin, 10, 12)
assert check_size(cos, 10, 12)
assert check_size(square, 10, 12)
assert check_size(sqrt, 10, 12)
assert check_size(tanh, 10, 12)
assert check_size(sigmoid, 10, 12)
assert check_size(reciprocal, 10, 12)
assert check_size(sum_, 10, 12)
assert check_size(subtract, 10, 12)
assert check_size(multiply, 10, 12)
assert check_size(divide, 10, 15)
assert check_size(minmaxnorm, 10, 12)
assert check_size(zscore, 10, 12)
assert check_size(max_, 10, 15)
assert check_size(min_, 10, 15)
assert check_size(std, 10, 15)
assert check_size(count, 10, 15)
assert check_size(mean, 10, 15)
assert check_size(z_agg, 10, 15)
assert check_size(datesplit, 10, 18)
assert check_size(union, 10, 15)

# print(df2[["c1__dummy__a", "c1__dummy__b", "c1__dummy__c"]])
# print(mean["mean_agg(n1, c1)"])

assert series_approx_equals(log["log n1"], [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302])
assert series_approx_equals(sin["sin n1"], [0.841, 0.909, 0.141, -0.756, -0.958, -0.279, 0.656, 0.989, 0.412, -0.544])
assert series_approx_equals(cos["cos n1"], [0.540, -0.416, -0.989, -0.653, 0.283, 0.960, 0.753, -0.145, -0.911, -0.839])
assert series_equals(square["square n1"], [1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
assert series_approx_equals(sqrt["sqrt n1"], [1, 1.414, 1.732, 2, 2.236, 2.449, 2.645, 2.828, 3, 3.162])
assert series_approx_equals(tanh["tanh n1"], [0.761, 0.964, 0.995, 0.999, 0.999, 1, 1, 1, 1, 1])
assert series_approx_equals(sigmoid["sigmoid n1"], [0.731, 0.880, 0.952, 0.982, 0.993, 0.997, 0.999, 1, 1, 1])
assert series_approx_equals(reciprocal["rc n1"], [1, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125, 0.111, 0.1])
assert series_equals(sum_["sum(n1, n2)"], [4, 8, 13, 4, 5, 11, 9, 5, 2, 11])
assert series_equals(subtract["subtract(n1, n2)"], [-2, -4, -7, 4, 5, 1, 5, 11, 16, 9])
assert series_equals(multiply["multiply(n1, n2)"], [3, 12, 30, 0, 0, 30, 14, -24, -63, 10])
assert series_approx_equals(divide["divide(n2, n1)"], [3, 3, 3.333, 0, 0, 0.833, 0.285, -0.375, -0.777, 0.1])
assert series_approx_equals(minmaxnorm["min_max_norm n1"], [0, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1])
assert series_approx_equals(zscore["zscore n1"], [-1.566, -1.218, -0.870, -0.522, -0.174, 0.174, 0.522, 0.870, 1.218, 1.566])
assert series_equals(max_["max_agg(n1, c1)"], [7, 10, 5, 5, 5, 10, 7, 10, 10, 10])
assert series_equals(min_["min_agg(n1, c1)"], [1, 2, 3, 3, 3, 2, 1, 2, 2, 2])
assert series_approx_equals(std["std_agg(n1, c1)"], [3, 2.828, 0.816, 0.816, 0.816, 2.828, 3, 2.828, 2.828, 2.828])
assert series_equals(count["count_agg(n1, c1)"], [2, 5, 3, 3, 3, 5, 2, 5, 5, 5])
assert series_equals(mean["mean_agg(n1, c1)"], [4, 7, 4, 4, 4, 7, 4, 7, 7, 7])
assert series_approx_equals(z_agg["z_agg_agg(n1, c1)"], [-1, -1.767, -1.224, 0, 1.224, -0.353, 1, 0.353, 0.707, 1.060])
assert series_approx_equals(datesplit["date_inferred_date_day"], [22, 22, 13, 1, 3, 4, 22, 22, 3, 3])
assert series_approx_equals(union["n1"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
assert series_approx_equals(union["n2"], [3, 6, 10, 0, 0, 5, 2, -3, -7, 1])

# Test applyBulkTransform

bulk_union = applyBulkTransform(df2, op=UnionOperation.UnionOperation(), firstPathsFromRoot=g.getPathsFromRootToNode(6),
                          secondPathsFromRoot=g.getPathsFromRootToNode(7), nodeOpDict=g.getOpsUsedToCreate())
bulk_log = applyBulkTransform(df2, op=OneArgOperation.LogOperation(), firstPathsFromRoot=g.getPathsFromRootToNode(4),
                              nodeOpDict=g.getOpsUsedToCreate())

assert check_size(bulk_log, 10, 33)
assert check_size(bulk_union, 10, 48)

print("\nALL TESTS PASSED!")

