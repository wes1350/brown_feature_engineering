class Operation:
    def __init__(self):
        self.operation = None
        self.opType = None

    def getOperation(self):
        return self.operation

    def isFeatureSelector(self):
        if self.opType is None:
            return False
        return self.opType == "feature_selection"

    def isUnion(self):
        if self.opType is None:
            return False
        return self.opType == "union"

    def isTwoArg(self):
        if self.opType is None:
            return False
        return self.opType == "two_arg"

    def isAgg(self):
        if self.opType is None:
            return False
        return self.opType == "aggregate"

    def transform(self, df):
        return None

    def is_redundant_to_self(self):
        """
        If True, means that we don't want to apply this operation to itself. E.g. don't split dates in a chain.
        Default here is False, but can override to True based on operation type.
        :return:
        """
        return False