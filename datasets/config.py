import json
import pickle
import numpy as np
import os

class ModelConfig():
    def __init__(self):

        self.h = 224
        self.w = 224
        self.neg_count = 4
        self.hidden_size = 1000
        self.dropout = 0.5

class GeMConfig():
    def __init__(self):

        self.neg_count = 4
        self.hidden_size = 512
        self.progress = True
        self.scale_list=[0.5, 0.7071, 1, 1.4147, 2.0]





