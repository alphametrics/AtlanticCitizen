"""
"""

import numpy as np

class CasinoState(object):
    def __init__(self, randomState=None):
        if randomState is None:
            randomState = np.random.RandomState(None)
        self.randomState = randomState
    def getRandomState(self):
        return self.randomState

INIT_SIMPLE_CASINO_STATE = CasinoState()

class CasinoParam(object):
    def __init__(self):
        pass
    def genSample(self, casinoState, size):
        raise TypeError("Pure Abstract Method")

EMPTY_CASINO_PARAM = CasinoParam()

class CasinoGame(object):
    def __init__(self, casinoParam=EMPTY_CASINO_PARAM, casinoState=INIT_SIMPLE_CASINO_STATE):
        self.casinoParam = casinoParam
        self.casinoState = casinoState
        
    def genSample(self, size):
        return self.casinoParam.genSample(self.casinoState, size)
    
    def evaluatePayoff(self, testSampleRet, decision):
        raise TypeError("Pure Abstract Method")

