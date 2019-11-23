from feature_engineering.data_transformation import DataframeTransform

import pandas as pd
from d3m import container


n1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n2 = pd.Series([3, 6, 10, 0, 0, 5, 2, -3, -7, 1])
n3 = pd.Series([77, 54, 124, -665, 104, 7, 16.3, 8.22, 19.102, 5])
c1 = pd.Series(['a', 'b', 'c', 'c', 'c', 'b', 'a', 'b', 'b', 'b'])
c2 = pd.Series(['X', 'Y', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'X', 'Y'])
dates = pd.Series(['06/22/18', '05/22/18', '03/13/17', '12/01/19', '09/03/07',
                  '09/04/19', '03/22/18', '05/22/18', '03/03/17', '03/03/16'])

# Note: str_col is a high cardinality categorical variable, is removed and re-added at the end
df1 = container.DataFrame(data=pd.DataFrame({"n1": n1, "n2": n2, "n3": n3, "c1": c1, "c2": c2,
                    "str_col": pd.Series([str(i) for i in range(10)])}))

df_date = container.DataFrame(data=pd.DataFrame({"n1": n1, "c1": c1,
                                                    "str_col": pd.Series([str(i) for i in range(10)]), "dates": dates}))

import unittest

class TestTransform(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_log(self):
        op_name = "log"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]", "opt_outs":
                       "[\"skip_remove_high_cardinality_cat_vars\"]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_sin(self):
        op_name = "sin"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0.841, 0.909, 0.141, -0.756, -0.958, -0.279, 0.656, 0.989, 0.412, -0.544]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_cos(self):
        op_name = "cos"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0.540, -0.416, -0.989, -0.653, 0.283, 0.960, 0.753, -0.145, -0.911, -0.839]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_square(self):
        op_name = "square"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_sqrt(self):
        op_name = "sqrt"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [1, 1.414, 1.732, 2, 2.236, 2.449, 2.645, 2.828, 3, 3.162]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_tanh(self):
        op_name = "tanh"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0.761, 0.964, 0.995, 0.999, 0.999, 1, 1, 1, 1, 1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_sigmoid(self):
        op_name = "sigmoid"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0.731, 0.880, 0.952, 0.982, 0.993, 0.997, 0.999, 1, 1, 1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_rc(self):
        op_name = "rc"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [1, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125, 0.111, 0.1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_sum(self):
        op_name = "sum"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "(n1, n2)", result.columns)
        self.assertIn(op_name + "(n1, n3)", result.columns)
        self.assertIn(op_name + "(n2, n3)", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + "(n1, n2)"])

        ans = [4, 8, 13, 4, 5, 11, 9, 5, 2, 11]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_subtract(self):
        op_name = "subtract"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "(n1, n2)", result.columns)
        self.assertIn(op_name + "(n1, n3)", result.columns)
        self.assertIn(op_name + "(n2, n3)", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + "(n1, n2)"])

        ans = [-2, -4, -7, 4, 5, 1, 5, 11, 16, 9]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_multiply(self):
        op_name = "multiply"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "(n1, n2)", result.columns)
        self.assertIn(op_name + "(n1, n3)", result.columns)
        self.assertIn(op_name + "(n2, n3)", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + "(n1, n2)"])

        ans = [3, 12, 30, 0, 0, 30, 14, -24, -63, 10]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_divide(self):
        op_name = "divide"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "(n1, n2)", result.columns)
        self.assertIn(op_name + "(n1, n3)", result.columns)
        self.assertIn(op_name + "(n2, n3)", result.columns)
        self.assertIn(op_name + "(n2, n1)", result.columns)
        self.assertIn(op_name + "(n3, n1)", result.columns)
        self.assertIn(op_name + "(n3, n2)", result.columns)
        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "(n2, n1)"])

        ans = [3, 3, 3.333, 0, 0, 0.833, 0.285, -0.375, -0.777, 0.1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_min_max_norm(self):
        op_name = "min_max_norm"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [0, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_zscore(self):
        op_name = "zscore"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n1"])

        ans = [-1.566, -1.218, -0.870, -0.522, -0.174, 0.174, 0.522, 0.870, 1.218, 1.566]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_max(self):
        op_name = "max"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]", "opt_outs":
                       "[\"skip_remove_high_cardinality_cat_vars\"]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)
        self.assertIn(op_name + "_agg(n1, str_col)", result.columns)
        self.assertIn(op_name + "_agg(n2, str_col)", result.columns)
        self.assertIn(op_name + "_agg(n3, str_col)", result.columns)

        self.assertEqual(len(result.columns), 15)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [7, 10, 5, 5, 5, 10, 7, 10, 10, 10]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_min(self):
        op_name = "min"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)

        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [1, 2, 3, 3, 3, 2, 1, 2, 2, 2]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_std(self):
        op_name = "std"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}",
                       "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)

        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [3, 2.828, 0.816, 0.816, 0.816, 2.828, 3, 2.828, 2.828, 2.828]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_count(self):
        op_name = "count"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}",
                       "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)

        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [2, 5, 3, 3, 3, 5, 2, 5, 5, 5]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_mean(self):
        op_name = "mean"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}",
                       "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)

        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [4, 7, 4, 4, 4, 7, 4, 7, 7, 7]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_z_agg(self):
        op_name = "z_agg"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}",
                       "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + "_agg(n1, c1)", result.columns)
        self.assertIn(op_name + "_agg(n1, c2)", result.columns)
        self.assertIn(op_name + "_agg(n2, c1)", result.columns)
        self.assertIn(op_name + "_agg(n2, c2)", result.columns)
        self.assertIn(op_name + "_agg(n3, c1)", result.columns)
        self.assertIn(op_name + "_agg(n3, c2)", result.columns)

        self.assertEqual(len(result.columns), 12)
        result = list(result[op_name + "_agg(n1, c1)"])

        ans = [-1, -1.767, -1.224, 0, 1.224, -0.353, 1, 0.353, 0.707, 1.060]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_date_split(self):
        op_name = "date_split"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}",
                       "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df_date.copy()).value
        self.assertIn("dates_inferred_date_day", result.columns)
        self.assertEqual(len(result.columns), 14)
        result = list(result["dates_inferred_date_day"])

        ans = [22, 22, 13, 1, 3, 4, 22, 22, 3, 3]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i])

    def test_union(self):
        op_name = "union"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"" + op_name + "\"}",
                       "paths": "[[0, 1, 3], [0, 2, 3]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn("log n1", result.columns)
        self.assertIn("rc n1", result.columns)
        self.assertEqual(len(result.columns), 12)
        result = list(result["log n1"])

        ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        result = list(result["rc n1"])

        ans = [1, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125, 0.111, 0.1]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_binning_u(self):
        op_name = "binning_u"
        hyperparams = {"operations": "{\"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n3"])

        ans = [9, 9, 9, 0, 9, 8, 8, 8, 8, 8]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_binning_d(self):
        pass

    def test_one_term_frequency(self):
        op_name = "one_term_frequency"
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\": \"" + op_name + "\"}", "paths": "[[0, 1]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn(op_name + " n1", result.columns)
        self.assertIn(op_name + " n2", result.columns)
        self.assertIn(op_name + " n3", result.columns)
        self.assertEqual(len(result.columns), 9)
        result = list(result[op_name + " n2"])

        ans = [1, 1, 1, 2, 2, 1, 1, 1, 1, 1]

        for i in range(len(ans)):
            self.assertEqual(result[i], ans[i])

    def test_compact_one_hot(self):
        pass

    def test_invalid_operation(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"stick\", \"2\":\"rc\", \"3\": \"union\"}",
                       "paths": "[[0, 1, 3], [0, 2, 3]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        with self.assertRaises(Exception):
            result = primitive.produce(inputs=df1.copy()).value

    def test_invalid_path_different_ends(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"union\"}",
                       "paths": "[[0, 1], [0, 2, 3]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        with self.assertRaises(Exception):
            result = primitive.produce(inputs=df1.copy()).value

    def test_invalid_path_nonexistent_node(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"union\"}",
                       "paths": "[[0, 1, 4], [0, 2, 4]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        with self.assertRaises(Exception):
            result = primitive.produce(inputs=df1.copy()).value

    def test_invalid_path_nonexistent_path(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"union\"}",
                       "paths": "[[0, 2], [0, 1, 2]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        with self.assertRaises(Exception):
            result = primitive.produce(inputs=df1.copy()).value

    def test_select_names(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"union\"}",
                       "paths": "[[0, 1, 3], [0, 2, 3]]",
                       "names_to_keep": "[\"n1\", \"n2\", \"rc n3\", \"log n1\"]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value

        self.assertIn("log n1", result.columns)
        self.assertIn("rc n3", result.columns)
        # 4 kept numeric + c1, c2, str_col + adding back n3 since it is one of the original columns
        self.assertEqual(len(result.columns), 8)
        result = list(result["log n1"])

        ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

    def test_multistep_path(self):
        hyperparams = {"operations": "{\"0\": \"INIT\", \"1\":\"log\", \"2\":\"rc\", \"3\": \"sum\"}",
                       "paths": "[[0, 1, 2, 3]]"}

        primitive = DataframeTransform(hyperparams=hyperparams)
        result = primitive.produce(inputs=df1.copy()).value
        self.assertIn("log n3", result.columns)
        self.assertIn("rc n1", result.columns)
        self.assertIn("rc log n1", result.columns)
        self.assertIn("sum(rc n1, rc log n2)", result.columns)
        self.assertIn("sum(n2, log n3)", result.columns)
        # 3 (+ 2) - > 6 (+ 2) -> 12 (+ 2) -> 78 (+ 2)
        self.assertEqual(len(result.columns), 81)
        result = list(result["log n1"])

        ans = [0, 0.693, 1.098, 1.386, 1.609, 1.791, 1.945, 2.079, 2.197, 2.302]

        for i in range(len(ans)):
            self.assertAlmostEqual(result[i], ans[i], delta=0.001)

if __name__ == '__main__':
    unittest.main()
