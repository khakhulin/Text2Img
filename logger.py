import numpy as np
import pandas as pd

class Logger:
    def __init__(self, fields):
        self.fields = fields
        self.readings = {
            name: [] for name in self.fields
        }
    
    def write(self, *values):
        assert len(values) == len(self.fields), 'Incorrect number of fields'
        
        for i, val in enumerate(values):
            self.readings[self.fields[i]].append(val)

    def to_csv(self, path):
        df = pd.DataFrame.from_dict(self.readings)
        df.to_csv(path, index=False)
