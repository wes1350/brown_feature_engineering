from feature_engineering.data_transformation import DataframeTransform

import numpy as np
import pandas as pd
from d3m import container

n1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n2 = pd.Series([3, 6, 10, 0, 0, 5, 2, -3, -7, 1])
n3 = pd.Series([77, 54, 124, -665, 104, 7, 16.3, 8.22, 19.102, 5])
c1 = pd.Series(['a', 'b', 'c', 'c', 'c', 'b', 'a', 'b', 'b', 'b'])
c2 = pd.Series(['X', 'Y', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'X', 'Y'])
c3 = pd.Series([str(i) for i in range(10)])
dates = pd.Series(['06/22/18', '05/22/18', '03/13/17', '12/01/19', '09/03/07',
                  '09/04/19', '03/22/18', '05/22/18', '03/03/17', '03/03/16'])

df1 = container.DataFrame(data=pd.DataFrame({"n1": n1, "n2": n2, "n3": n3, "c1": c1, "c2": c2, "c3": c3}))
df_date = container.DataFrame(data=pd.DataFrame({"n1": n1, "c1": c1, "c3": c3, "dates": dates}))

import unittest

class TestTransform(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Operation tests by type

    def test_one_arg(self):
        answers_by_operation = {
            "log": [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302],
            "sin": [0.841, 0.909, 0.141, -0.756, -0.958, -0.279, 0.656, 0.989, 0.412, -0.544],
            "cos": [0.540, -0.416, -0.989, -0.653, 0.283, 0.960, 0.753, -0.145, -0.911, -0.839],
            "square": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
            "sqrt": [1, 1.414, 1.732, 2, 2.236, 2.449, 2.645, 2.828, 3, 3.162],
            "rc": [0.761, 0.964, 0.995, 0.999, 0.999, 1, 1, 1, 1, 1],
            "sigmoid": [0.731, 0.880, 0.952, 0.982, 0.993, 0.997, 0.999, 1, 1, 1],
            "tanh": [1, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125, 0.111, 0.1]
        }
        for op_name in answers_by_operation:
            hyperparams = {"features": ["{op}(n{c})".format(op=op_name, c=str(i)) for i in [1, 2, 3]]}

            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
            for feature in hyperparams["features"]:
                self.assertIn(feature, result.columns)
            self.assertEqual(len(result.columns), df1.shape[1] + 3)
            result_column = list(result["{op}(n1)".format(op=op_name)])

            ans = answers_by_operation[op_name]
            for i in range(len(ans)):
                self.assertAlmostEqual(result_column[i], ans[i], delta=0.001)

    def test_two_arg(self):
        answers_by_operation = {
            "sum": [4, 8, 13, 4, 5, 11, 9, 5, 2, 11],
            "subtract": [-2, -4, -7, 4, 5, 1, 5, 11, 16, 9],
            "multiply": [3, 12, 30, 0, 0, 30, 14, -24, -63, 10],
            "divide": [3, 3, 3.333, 0, 0, 0.833, 0.285, -0.375, -0.777, 0.1]
        }

        for op_name in answers_by_operation:
            hyperparams = {"features": ["{op}(n{i}, n{j})".format(op=op_name, i=c%3+1, j=c/3+1) for c in range(3**2) if
                                        c%3 != c/3]}

            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
            for feature in hyperparams["features"]:
                self.assertIn(feature, result.columns)
            self.assertEqual(len(result.columns), df1.shape[1] + 6)
            result_column = list(result["{op}(n1, n2)".format(op=op_name)])

            ans = answers_by_operation[op_name]
            for i in range(len(ans)):
                self.assertAlmostEqual(result_column[i], ans[i], delta=0.001)

    def test_statistical(self):
        answers_by_operation = {
            "zscore": [-1.566, -1.218, -0.870, -0.522, -0.174, 0.174, 0.522, 0.870, 1.218, 1.566],
            "min_max_norm": [0, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1],
            "binning_u": [9, 9, 9, 0, 9, 8, 8, 8, 8, 8]
        }
        for op_name in answers_by_operation:
            hyperparams = {"features": ["{op}(n{c})".format(op=op_name, c=str(i)) for i in [1, 2, 3]]}

            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
            for feature in hyperparams["features"]:
                self.assertIn(feature, result.columns)
            self.assertEqual(len(result.columns), df1.shape[1] + 3)
            result_column = list(result["{op}(n1)".format(op=op_name)])

            ans = answers_by_operation[op_name]
            for i in range(len(ans)):
                self.assertAlmostEqual(result_column[i], ans[i], delta=0.001)

    def test_aggregate(self):
        answers_by_operation = {
            "max_agg": [7, 10, 5, 5, 5, 10, 7, 10, 10, 10],
            "min_agg": [1, 2, 3, 3, 3, 2, 1, 2, 2, 2],
            "count_agg": [2, 5, 3, 3, 3, 5, 2, 5, 5, 5],
            "mean_agg": [4, 7, 4, 4, 4, 7, 4, 7, 7, 7],
            "std_agg": [3, 2.828, 0.816, 0.816, 0.816, 2.828, 3, 2.828, 2.828, 2.828],
            "zscore_agg": [-1, -1.767, -1.224, 0, 1.224, -0.353, 1, 0.353, 0.707, 1.060]
        }

        for op_name in answers_by_operation:
            hyperparams = {"features": ["{op}(n{n}, c{c})".format(op=op_name, n=i%3+1, c=i/3+1) for i in range(3**2)]}

            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
            for feature in hyperparams["features"]:
                self.assertIn(feature, result.columns)
            self.assertEqual(len(result.columns), df1.shape[1] + 3**2)
            result_column = list(result["{op}(n1, c1)".format(op=op_name)])

            ans = answers_by_operation[op_name]
            for i in range(len(ans)):
                self.assertAlmostEqual(result_column[i], ans[i], delta=0.001)

    def test_date_split_specific(self):
        op_name = "date_split"
        hyperparams = {"features": ["{op}_{time}(dates)".format(op=op_name, time=t) for t in ["year, hour, day, microsecond"]]}
        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df_date).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df_date.shape[1] + 4)
        result_column = list(result["{op}_day".format(op=op_name)])

        ans = [22, 22, 13, 1, 3, 4, 22, 22, 3, 3]
        for i in range(len(ans)):
            self.assertEqual(result_column[i], ans[i])

    def test_date_split_full(self):
        op_name = "date_split"
        hyperparams = {"features": ["{op}(dates)".format(op=op_name)]}
        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df_date).value
        self.assertEqual(len(result.columns), df_date.shape[1] + 9)

    def test_one_term_frequency(self):
        op_name = "one_term_frequency"
        hyperparams = {"features": ["{op}(n{c})".format(op=op_name, c=str(i)) for i in [1, 2, 3]]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + 3)
        result_column = list(result["{op}(n2)".format(op=op_name)])

        ans = [1, 1, 1, 2, 2, 1, 1, 1, 1, 1]
        for i in range(len(ans)):
            self.assertEqual(result_column[i], ans[i])

    # Testing production with multiple operations

    def test_multiple_operations(self):
        answers_by_operation = {
            "log": [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302],
            "sin": [0.841, 0.909, 0.141, -0.756, -0.958, -0.279, 0.656, 0.989, 0.412, -0.544],
        }

        hyperparams = {"features": ["{op}(n1)".format(op=op) for op in answers_by_operation]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + len(answers_by_operation))

        for op_name in answers_by_operation:
            result_column = list(result["{op}(n1)".format(op=op_name)])
            ans = answers_by_operation[op_name]
            for i in range(len(ans)):
                self.assertAlmostEqual(result_column[i], ans[i], delta=0.001)

    # Test for composed operations

    def test_composed_operation(self):
        log_ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]
        log_log_ans = [(lambda x: (0 if x <= 0 else np.log(x)))(v) for v in log_ans]

        hyperparams = {"features": ["log(n1)", "log(log(n1))", "log(log(n2))"]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + len(hyperparams["features"]))

        result_column_composed = list(result["log(log(n1))"])
        result_column_original = list(result["log(n1)"])
        for i in range(len(log_ans)):
            self.assertAlmostEqual(result_column_composed[i], log_log_ans[i], delta=0.001)
            self.assertAlmostEqual(result_column_original[i], log_ans[i], delta=0.001)


    def test_composed_operation_mixed(self):
        log_ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]
        sin_log_ans = [np.sin(v) for v in log_ans]

        hyperparams = {"features": ["log(n1)", "sin(log(n1))", "sin(log(n2))"]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + len(hyperparams["features"]))

        result_column_composed = list(result["sin(log(n1))"])
        result_column_original = list(result["log(n1)"])
        for i in range(len(log_ans)):
            self.assertAlmostEqual(result_column_composed[i], sin_log_ans[i], delta=0.001)
            self.assertAlmostEqual(result_column_original[i], log_ans[i], delta=0.001)

    # Test on large set of features to generate

    def test_large_number_of_features(self):
        df = pd.DataFrame({})
        for i in range(100):
            df["column_{n}".format(n=i)] = range(i, i+10)
        hyperparams = {"features": ["sum(column_{i}, column_{j})".format(i=str(k/100+1), j=str(k%100+1))
                                    for k in range(100**2) if k/100 != k%100]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + len(hyperparams["features"]))

        for i in range(100):
            for j in range(100):
                for k in range(10):
                    self.assertEqual(result["sum(column_{i}, column_{j}".format(i=str(i+1), j=str(j+1))],
                                     df["column_{i}".format(i=str(i+1))]+df["column_{j}".format(j=str(j+1))])

    # Test for invalid operations

    def test_invalid_operation(self):
        hyperparams = {"features": ["invalid(n1)"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    def test_invalid_operation_mixed_with_valid(self):
        hyperparams = {"features": ["log(n1), sin(n1), invalid(n1), sum(n1, n2)"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    # Test valid features, but invalid parentheses

    def test_invalid_parenthesis_left(self):
        hyperparams = {"features": ["invalid((n1)"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    def test_invalid_parenthesis_right(self):
        hyperparams = {"features": ["invalid(n1))"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    def test_invalid_parenthesis_extra_pair(self):
        hyperparams = {"features": ["invalid(n1)()"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    def test_invalid_parentheses_mixed_with_valid(self):
        hyperparams = {"features": ["log(n1), sin(n1), rc((n1), sum(n1, n2)"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    # Test for invalid column

    def test_invalid_column(self):
        hyperparams = {"features": ["log(nonexistent_column)"]}

        with self.assertRaises(Exception):
            result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value

    # Test for duplicate column

    def test_duplicate_column(self):
        hyperparams = {"features": ["log(n1), rc(n1), log(n1)"]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + 3 - 1)

    # Test for including original features

    def test_with_original_features(self):
        op_name = "log"
        hyperparams = {"features": ["{op}(n{c})".format(op=op_name, c=str(i)) for i in [1, 2, 3]] + ["n1", "n2"]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        for feature in hyperparams["features"]:
            self.assertIn(feature, result.columns)
        self.assertEqual(len(result.columns), df1.shape[1] + 3)

    # Test for no new features in list

    def test_empty_list(self):
        hyperparams = {"features": []}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        self.assertEqual(len(result.columns), df1.shape[1])

    def test_no_new_features_but_originals_included(self):
        hyperparams = {"features": ["n1", "n2"]}

        result = DataframeTransform(hyperparams=hyperparams).produce(inputs=df1).value
        self.assertEqual(len(result.columns), df1.shape[1])

if __name__ == '__main__':
    unittest.main()
