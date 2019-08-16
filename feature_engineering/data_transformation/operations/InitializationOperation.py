from .Operation import Operation

class InitializationOperation(Operation):
    def __init__(self):
        super().__init__()
        self.operation = "INIT"
        self.opType = "Init"


