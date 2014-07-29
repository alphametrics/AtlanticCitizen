from patrons.base import *

import scipy as sp
from scipy import linalg

#import statsmodels.api as sm

class UncertaintyAwarePatronParam(PatronParam):
    def __init__(self, scaleFactor=1, cost=0.0):
        self.scaleFactor = scaleFactor
        self.cost = cost
        
    def calcQuantityBet(self, alphaHat, a1dUncertainty):
        qtyBet = self.scaleFactor * (alphaHat - self.cost) / a1dUncertainty
        qtyBet[qtyBet < 0] = 0
        
        return qtyBet

class UncertaintyAwarePatronState(PatronState):
    def __init__(self, betaHat=None, precMat=None):
        self.betaHat = betaHat
        self.precMat = precMat
        PatronState.__init__(self)
    def update(self, newSample):
        a1dRet, a2dPred = newSample
        nObs, nPred = a2dPred.shape
        assert a1dRet.shape == (nObs,)
        
        gramMat = np.dot(a2dPred.T, a2dPred)
        a1dEigVals = np.zeros((nPred,), dtype=np.double)
        a2dEigVecs = np.zeros((nPred, nPred), dtype=np.double)
        #a1dEigVals[::-1], a2dEigVecs[::-1] = linalg.eigh(gramMat, overwrite_a=True)
        a1dEigVals, a2dEigVecs = linalg.eigh(gramMat, overwrite_a=True)
        
        lstsqRes = linalg.lstsq(a2dPred, a1dRet)
        
        if False:
            olsModel = sm.OLS(a1dRet, a2dPred)
            olsFit = olsModel.fit()
            print olsFit.summary()
        
        self.betaHat = lstsqRes[0]
        self.eigVals = 1.0 / a1dEigVals
        self.sqrtEigVals = np.sqrt(self.eigVals)
        self.eigVecs = a2dEigVecs
        self.projMat = self.eigVecs * np.diag(self.sqrtEigVals)
        
    def calcUncertainty(self, testSamplePred):
        nTest, nPred = testSamplePred.shape
        a2dPred = testSamplePred
        
        a1dProjLoadings = np.dot(a2dPred, self.projMat)
        assert a1dProjLoadings.shape == (nTest, nPred)
        
        a1dUncertainty = np.sum(a1dProjLoadings * a1dProjLoadings, axis=1)
        assert a1dUncertainty.shape == (nTest,)
        return a1dUncertainty


class UncertaintyAwarePatronPolicy(PatronPolicy):
    def __init__(self, patronParam=PatronParam(), patronState=UncertaintyAwarePatronState()):
        self.patronParam = patronParam
        self.patronState = patronState
        
    def trainPatronPolicy(self, sample):
        self.patronState.update(sample)
        
    def applyPatronPolicy(self, testSamplePred):
        a2dPred = testSamplePred
        
        alphaHat = np.dot(a2dPred, self.patronState.betaHat)
        a1dUncertainty = self.patronState.calcUncertainty(testSamplePred)
        
        qtyBet = self.patronParam.calcQuantityBet(alphaHat, a1dUncertainty)
        
        return qtyBet

