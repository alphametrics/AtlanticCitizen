
import casinos.linearGaussianCasino as lgc
import patrons.linearGaussianCasino as lgp

import numpy as np

import matplotlib.pyplot as plt

def runTrial(randomState, scalarSigma, betaTrue, meanPred, covPred \
             , sizeTrain, sizeTest \
             , cost \
             ):
    linearGaussianCasinoParam = lgc.LinearGaussianCasinoParam(
            scalarSigma \
            , betaTrue \
            , meanPred \
            , covPred \
            )
    linearGaussianCasinoParam.setCost(cost)
    
    linearGaussianCasinoState = lgc.LinearGaussianCasinoState(randomState=randomState)
    
    linearGaussianCasinoGame = lgc.LinearGaussianCasinoGame(\
              casinoParam=linearGaussianCasinoParam \
            , casinoState=linearGaussianCasinoState \
            )
    
    naiveBluePatronParam = lgp.NaiveBluePatronParam(cost=cost)
    naiveBluePatronState = lgp.NaiveBluePatronState()
    
    uncertaintyAwarePatronParam = lgp.UncertaintyAwarePatronParam(cost=cost)
    uncertaintyAwarePatronState = lgp.UncertaintyAwarePatronState()
    
    ##
    
    naiveBluePatronPolicy = lgp.NaiveBluePatronPolicy(\
              patronParam=naiveBluePatronParam \
            , patronState=naiveBluePatronState \
            )
    uncertaintyAwarePatronPolicy = lgp.UncertaintyAwarePatronPolicy(\
              patronParam=uncertaintyAwarePatronParam \
            , patronState=uncertaintyAwarePatronState \
            )
    
    sampleTrain = linearGaussianCasinoGame.genSample(sizeTrain)
    sampleTest  = linearGaussianCasinoGame.genSample(sizeTest)
    trainSampleRet, trainSamplePred = sampleTrain
    testSampleRet, testSamplePred = sampleTest
    idxTestSample = np.arange(sizeTest)
    
    if False:
        fig = plt.subplot(1, 1, 1)
        bins = np.linspace(-15, 15, 30 / 0.1 + 1)
        fig.hist(trainSampleRet, bins=bins, histtype="step")
        fig.hist(testSampleRet, bins=bins, histtype="step")
        plt.show()
        
    naiveBluePatronPolicy.trainPatronPolicy(sampleTrain)
    uncertaintyAwarePatronPolicy.trainPatronPolicy(sampleTrain)
    
    if False:
        alphaTrueInTest = np.dot(testSamplePred, betaTrue)
        alphaHatInTestNaiveBlue         = np.dot(testSamplePred, naiveBluePatronState.betaHat)
        alphaHatInTestUncertaintyAware  = np.dot(testSamplePred, uncertaintyAwarePatronState.betaHat)
        plt.step(idxTestSample, alphaTrueInTest,                where="post", color="b")
        plt.step(idxTestSample, alphaHatInTestNaiveBlue,        where="post", color="g")
        plt.step(idxTestSample, alphaHatInTestUncertaintyAware, where="post", color="r")
        plt.show()

    
    qtyNaiveBlue        = naiveBluePatronPolicy.applyPatronPolicy(testSamplePred)
    qtyUncertaintyAware = uncertaintyAwarePatronPolicy.applyPatronPolicy(testSamplePred)
    
    pnlNaiveBlue        = linearGaussianCasinoGame.evaluatePayoff(testSampleRet, qtyNaiveBlue)
    pnlUncertaintyAware = linearGaussianCasinoGame.evaluatePayoff(testSampleRet, qtyUncertaintyAware)
    
    volNaiveBlue        = np.sqrt(np.mean(pnlNaiveBlue * pnlNaiveBlue))
    volUncertaintyAware = np.sqrt(np.mean(pnlUncertaintyAware * pnlUncertaintyAware))
    
    if False:
        plt.step(idxTestSample, np.cumsum(pnlNaiveBlue) / volNaiveBlue,        where="post", color="g")
        plt.step(idxTestSample, np.cumsum(pnlUncertaintyAware) / volUncertaintyAware, where="post", color="r")
        plt.show()
    
    return np.sum(pnlNaiveBlue), np.sum(pnlUncertaintyAware)


seed = 123
randomState = np.random.RandomState(seed=seed)

scalarSigma = 1.0
numPred = 10

rho = 0.1

meanPred    = np.zeros(numPred, dtype=np.double)
covPred     = np.ones((numPred, numPred), dtype=np.double) * rho
covPred     += np.eye(numPred) * (1.0 - rho)

assert np.all(np.diag(covPred) == 1.0)

sizeTrain   = 1000
sizeTest    = 1000

cost = 1.0

betaPriorMean   = np.zeros(numPred, dtype=np.double)
betaPriorCov    = np.ones((numPred, numPred), dtype=np.double) * rho
betaPriorCov    += np.eye(numPred) * (1.0 - rho)
assert np.all(np.diag(betaPriorCov) == 1.0)

betaPriorCov    *= 0.01

nTrials = 1000

idxDates = np.arange(nTrials)
dlyPnlNaiveBlue = np.zeros(nTrials, dtype=np.double)
dlyPnlUncertaintyAware = np.zeros(nTrials, dtype=np.double)

for iTrial in np.arange(nTrials):
    betaTrue = randomState.multivariate_normal(betaPriorMean, betaPriorCov)
    
    eodPnlNaiveBlue, eodPnlUncertaintyAware = \
    runTrial(randomState, scalarSigma, betaTrue, meanPred, covPred \
             , sizeTrain, sizeTest \
             , cost \
             )
    
    dlyPnlNaiveBlue[iTrial] = eodPnlNaiveBlue
    dlyPnlUncertaintyAware[iTrial] = eodPnlUncertaintyAware
    
    pass

if True:
    stdNaiveBlue = np.std(dlyPnlNaiveBlue)
    stdUncertaintyAware = np.std(dlyPnlUncertaintyAware)
    plt.step(idxDates, np.cumsum(dlyPnlNaiveBlue) / stdNaiveBlue,               where="post", color="g")
    plt.step(idxDates, np.cumsum(dlyPnlUncertaintyAware) / stdUncertaintyAware, where="post", color="r")
    plt.show()

pass
