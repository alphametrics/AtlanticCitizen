from patrons.base import *

import scipy as sp
from scipy import linalg

class NaiveBluePatronParam(PatronParam):
    def __init__(self, scaleFactor=1, cost=0.0):
        self.scaleFactor = scaleFactor
        self.cost = cost
    def calcQuantityBet(self, alphaHat):
        qtyBet = self.scaleFactor * (alphaHat - self.cost)
        qtyBet[qtyBet < 0] = 0
        return qtyBet

class NaiveBluePatronState(PatronState):
    def __init__(self, betaHat=None):
        self.betaHat = betaHat
        PatronState.__init__(self)
    def update(self, newSample):
        a1dRet, a2dPred = newSample
        nObs, nPred = a2dPred.shape
        assert a1dRet.shape == (nObs,)
        lstsqRes = linalg.lstsq(a2dPred, a1dRet)
        self.betaHat = lstsqRes[0]

class NaiveBluePatronPolicy(PatronPolicy):
    def __init__(self, patronParam=NaiveBluePatronParam(), patronState=NaiveBluePatronState()):
        self.patronParam = patronParam
        self.patronState = patronState
        
    def trainPatronPolicy(self, sample):
        self.patronState.update(sample)
        
    def applyPatronPolicy(self, testSamplePred):
        a2dPred = testSamplePred
        
        alphaHat = np.dot(a2dPred, self.patronState.betaHat)
        qtyBet = self.patronParam.calcQuantityBet(alphaHat)
        
        return qtyBet

