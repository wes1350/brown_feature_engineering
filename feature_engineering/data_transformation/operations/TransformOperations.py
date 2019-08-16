import numpy as np
import scipy.stats as st

# NOTE: If editing the given array, make sure to copy it first, so that we don't affect the original column.

def oneArgOperationWrapper(op_name, a):
    if op_name == "square":
        return np.square(a)
    elif op_name == "log":
        A = a.copy()
        A[A <= 0] = 1
        return np.log(A)
    elif op_name == "sin":
        return np.sin(a)
    elif op_name == "sqrt":
        A = a.copy()
        A[A < 0] = 0
        return np.sqrt(A)
    elif op_name == "cos":
        return np.cos(a)
    elif op_name == "rc":
        # np.reciprocal doesn't seem to work here. Note: np.reciprocal doesn't work with integers
        A = a.copy()
        zeros = A == 0
        A[zeros] = 1
        A = 1/A
        A[zeros] = 0
        return A
    elif op_name == "tanh":
        return np.tanh(a)
    elif op_name == "sigmoid":
        return 0.5*(1 + np.tanh(0.5*a))
    else:
        raise Exception("Invalid One Arg Operation: " + op_name)

def twoArgOperationWrapper(op_name, a, b):
    if op_name == "sum":
        return np.add(a, b)
    elif op_name == "subtract":
        return np.subtract(a, b)
    elif op_name == "multiply":
        return np.multiply(a, b)
    elif op_name == "divide":
        A = a.copy()
        B = b.copy()
        zeros = B == 0
        B[zeros] = 1
        A = A / B
        A[zeros] = 0
        return A
    else:
        raise Exception("Invalid Two Arg Operation: " + op_name)

def statisticalOperationWrapper(op_name, a):
    if op_name == "zscore":
        a_min = np.amin(a)
        a_max = np.amax(a)
        if a_min == a_max:
            return [0] * len(a)
        return st.zscore(a)
    elif op_name == "min_max_norm":
        a_min = np.amin(a)
        a_max = np.amax(a)
        if a_min == a_max:
            return [0]*len(a)
        A = a.copy()
        A = A - a_min
        return A / (a_max - a_min)
    elif op_name == "binning_u":
        a_min = np.amin(a)
        a_max = np.amax(a)
        if a_max == a_min:
            return np.ones(len(a))
        n_bins = 10#max(2, min(int(len(a)/5), 10))
        binned = np.zeros(len(a))
        for i in range(len(a)):
            # min ensures at max, we don't overflow
            binned[i] = min(np.floor((a[i] - a_min) / (a_max - a_min) * n_bins), n_bins - 1)
        return binned
    elif op_name == "binning_d":
        a_min = np.amin(a)
        a_max = np.amax(a)
        iqr = st.iqr(a)
        if iqr == 0: # this formula degenerates, and we get infs. Just return one bin as default
            return np.ones(len(a))
        n_bins = np.ceil((a_max - a_min) / (2 * iqr / (len(a) ** (1 / 3))))  # Freedman-Diaconis rule
        binned = np.zeros(len(a))
        for i in range(len(a)):
            # min ensures at max, we don't overflow
            binned[i] = min(np.floor((a[i] - a_min) / (a_max - a_min) * n_bins), n_bins - 1)
        return binned
    else:
        raise Exception("Invalid Statistical Operation: " + op_name)

def aggregateOperationWrapper(op_name, a):
    if len(a) == 0:
        raise Exception("Cannot compute aggregate operation of empty list of numeric attributes!")
    if op_name == "min" or op_name == "spatial_min":
        return np.amin(a)
    elif op_name == "max" or op_name == "spatial_max":
        return np.amax(a)
    elif op_name == "count" or op_name == "spatial_count":
        return len(a)
    elif op_name == "mean" or op_name == "spatial_mean":
        return np.mean(a)
    elif op_name == "std" or op_name == "spatial_std":
        a_min = np.amin(a)
        a_max = np.amax(a)
        if a_min == a_max or len(a) == 1:
            return 0
        return np.std(a)
    elif op_name == "z_agg" or op_name == "spatial_z_agg":
        a_min = np.amin(a)
        a_max = np.amax(a)
        if a_min == a_max or len(a) == 1:
            return 0
        return st.zscore(a)
    else:
        raise Exception("Invalid Aggregate Operation: " + op_name)