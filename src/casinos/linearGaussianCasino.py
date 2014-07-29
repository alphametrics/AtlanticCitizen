
from casinos.base import *

import numpy as np

class LinearGaussianCasinoState(CasinoState):
    def __init__(self, randomState=None):
        CasinoState.__init__(self, randomState)

class LinearGaussianCasinoParam(CasinoParam):
    def __init__(self, scalarSigma, betaTrue, meanPred, covPred):
        if not isinstance(betaTrue, np.ndarray) or betaTrue.ndim != 1 or not betaTrue.size > 0:
            raise ValueError("Invalid Input")
        
        CasinoParam.__init__(self)
        
        self.scalarSigma = scalarSigma
        self.betaTrue = betaTrue
        self.numPred = betaTrue.size
        
        self.meanPred   = meanPred
        self.covPred    = covPred
        
    def setCost(self, cost=0.0):
        self.cost = cost
        
    def getCost(self):
        return self.cost
        
    def genSample(self, casinoState, size):
        randomState = casinoState.getRandomState()
        
        a2dPred = randomState.multivariate_normal(self.meanPred, self.covPred, size)
        a1dIdio = randomState.normal(scale=self.scalarSigma, size=size)
        
        assert a2dPred.shape == (size, self.numPred)
        assert a1dIdio.shape == (size,)
        a1dRet  = np.dot(a2dPred, self.betaTrue) + a1dIdio
        
        return a1dRet, a2dPred


class LinearGaussianCasinoGame(CasinoGame):
    def __init__(self \
                 , casinoParam=None \
                 , casinoState=None \
                 ):
        CasinoGame.__init__(self, casinoParam, casinoState)
        
    def evaluatePayoff(self, testSampleRet, decision):
        if np.any(decision < 0):
            raise ValueError("Should only take the positive side of bets")
        cost = self.casinoParam.getCost()
        return (testSampleRet - cost) * decision

