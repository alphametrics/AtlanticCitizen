"""
"""

import numpy as np

class PatronState(object):
    def __init__(self):
        pass

STATELESS_PATRON_STATE = PatronState()

class PatronParam(object):
    def __init__(self):
        pass

EMPTY_Patron_PARAM = PatronParam()

class PatronPolicy(object):
    def __init__(self, patronParam=EMPTY_Patron_PARAM, patronState=STATELESS_PATRON_STATE):
        self.patronParam = patronParam
        self.patronState = patronState
        
    def trainPatronPolicy(self, sample):
        raise TypeError("Pure Abstract Method")
    def applyPatronPolicy(self, testSamplePred):
        raise TypeError("Pure Abstract Method")

