import json
import pickle
import numpy as np

class ModelConfig():
    def __init__(self):

        self.h = 224
        self.w = 224
        self.neg_count = 4
        self.hidden_size = 1000
        self.dropout = 0.5


