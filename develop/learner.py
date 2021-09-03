"""a more clean api using graph computation """
import numpy as np


"""
x -> L1 -> L2 -> yhat
grad -> dL2 -> dL1
"""


class Learner:
    """for better api, similar to model.compile() in tf"""
    # or just create a complete class called 'Model'
    # like tensorflow api

    def __init__(self, model, lossf, optimizer):
        """i don't care of using list but i guess i have to """
        # figure out how to assemble the layers
        # list.extend method could be useful
        self.model = model
        # loss function instead of the class
        self.lossf = lossf
        # function also
        self.optimizer = optimizer

    def __call__(self, x, y):
        """forward-backward"""
        yhat = self.model(x)
        self.lossf(yhat, y)
        self.optimizer()
