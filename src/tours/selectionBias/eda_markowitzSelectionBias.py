import numpy as np

import matplotlib.pyplot as plt

import common.kato_numpy as knp

binaryPolicy    = lambda f: (f > 0) * 1.0
hockeyPolicy    = lambda f: (f > 0) * f
quadraticPolicy = lambda f: (f > 0) * (f * f)

LIST_OF_METRICS = ["BetQty", "ExpectedPnl", "RawFcstPnl", "ModFcstPnl", "SelectionBias", "RetTotalVaria", "RetResidVaria"]
NAMES_OF_POLICIES = ["binary", "hockey", "quadratic"]

#LIST_OF_COLORS = ["", "", ""]

def getSizingPolicy(name):
    return eval(name + "Policy")

square = lambda a: a * a

def stdErrMean(a):
    return np.std(a, ddof=1) / np.sqrt(a.size)

def constructBayesianShrinkageOperator(meanAlphaPrior, stdAlphaPrior, stdErrRawFcst, naivePolicy):
    precAlphaPrior  = 1. / (stdAlphaPrior * stdAlphaPrior)
    precRawFcst     = 1. / (stdErrRawFcst * stdErrRawFcst)
    blendWgtOnPrior = precAlphaPrior / (precAlphaPrior + precRawFcst)
    blendWgtOnRaw   = precRawFcst / (precAlphaPrior + precRawFcst)
    
    def calcBayesianShrunkFcst(f):
        return blendWgtOnPrior * meanAlphaPrior + f * blendWgtOnRaw
    
    def bayesianShrunkPolicy(f):
        return naivePolicy(calcBayesianShrunkFcst(f))
    
    return bayesianShrunkPolicy


def calibGammaFcstPrecision(variaRawFcst, numFeatures=10):
    shapePrecRawFcst = numFeatures / 2.0
    
    #scalePrecRawFcst = np.sqrt(variaRawFcst / shapePrecRawFcst)
    scalePrecRawFcst = 1.0 / np.sqrt(variaRawFcst * square(shapePrecRawFcst - 1) * (shapePrecRawFcst - 2))
    
    equivChiSqStdOfEachDof = 2. / scalePrecRawFcst
    
    return shapePrecRawFcst, scalePrecRawFcst


def genGaussianPriorGammaFcstPrecisionExperiments(seed, size, meanAlphaPrior, stdAlphaPrior \
            , shapePrecRawFcst, scalePrecRawFcst \
            , constNoiseRet=0 \
            , sizingPolicy=binaryPolicy \
            , applyBayesianShrinkage=False \
            ):
    
    randomState = np.random.RandomState(seed=seed)
    
    sampAlphaTrue       = randomState.normal(loc=meanAlphaPrior, scale=stdAlphaPrior, size=size)
    sampPrecRawFcst     = randomState.gamma(shape=shapePrecRawFcst, scale=scalePrecRawFcst, size=size)
    sampStdErrRawFcst   = np.sqrt(1.0 / sampPrecRawFcst)
    
    sampAlphaHatRaw     = randomState.normal(loc=0.0, scale=1.0, size=size) * sampStdErrRawFcst + sampAlphaTrue
    
    sampAssetRet        = randomState.normal(loc=0.0, scale=1.0, size=size) * constNoiseRet + sampAlphaTrue
    
#    finalSizingPolicy = sizingPolicy
#    if applyBayesianShrinkage:
#        finalSizingPolicy = constructBayesianShrinkageOperator(meanAlphaPrior, stdAlphaPrior \
#                , sampStdErrRawFcst, sizingPolicy \
#                )
    
    if applyBayesianShrinkage:
        precAlphaPrior  = 1. / (stdAlphaPrior * stdAlphaPrior)
        #precRawFcst     = 1. / (sampStdErrRawFcst * sampStdErrRawFcst)
        precRawFcst     = sampPrecRawFcst.copy()
        assert np.all(np.abs(precRawFcst - 1. / (sampStdErrRawFcst * sampStdErrRawFcst)) < 1E-12)
        blendWgtOnPrior = precAlphaPrior / (precAlphaPrior + precRawFcst)
        blendWgtOnRaw   = precRawFcst / (precAlphaPrior + precRawFcst)
        
        sampAlphaHatMod = blendWgtOnPrior * meanAlphaPrior + blendWgtOnRaw * sampAlphaHatRaw
    else:
        sampAlphaHatMod = sampAlphaHatRaw.copy()
    
    sampBetQty = sizingPolicy(sampAlphaHatMod)
    
    return sampBetQty, sampAlphaTrue, sampAlphaHatRaw, sampAlphaHatMod, sampAssetRet


def summarizeGaussianPriorGammaFcstPrecisionExperiments(seed, size, meanAlphaPrior, stdAlphaPrior \
            , shapePrecRawFcst, scalePrecRawFcst \
            , constNoiseRet=0 \
            , sizingPolicy=binaryPolicy \
            , applyBayesianShrinkage=False \
            ):
    
    sampBetQty, sampAlphaTrue, sampAlphaHatRaw, sampAlphaHatMod, sampAssetRet = \
    genGaussianPriorGammaFcstPrecisionExperiments(seed, size, meanAlphaPrior, stdAlphaPrior \
            , shapePrecRawFcst, scalePrecRawFcst \
            , constNoiseRet=constNoiseRet \
            , sizingPolicy=sizingPolicy \
            , applyBayesianShrinkage=applyBayesianShrinkage \
            )
    
    assert applyBayesianShrinkage or np.all(sampAlphaHatMod == sampAlphaHatRaw)
    
    sampExpectedPnl     = sampBetQty * sampAlphaTrue
    sampRawFcstPnl      = sampBetQty * sampAlphaHatRaw
    sampModFcstPnl      = sampBetQty * sampAlphaHatMod
    sampSelectionBias   = sampExpectedPnl - sampModFcstPnl
    sampRetTotalVaria   = square(sampBetQty * sampAssetRet)
    sampRetResidVaria   = square(sampBetQty * (sampAssetRet - sampAlphaHatMod))
    
    res = dict()
    
    def populateKeyMetrics(res, key, sampArray):
        res["mean" + key]   = np.mean(sampArray)
        res["stdErr" + key] = np.std(sampArray, ddof=1) / np.sqrt(sampArray.size)
    
    for key in LIST_OF_METRICS:
        populateKeyMetrics(res, key, eval("samp" + key))
    
    return res


constNoiseRet = 100

variaRawFcstUncond = square(constNoiseRet) * 0.02

meanAlphaPrior = -2.0 * np.sqrt(variaRawFcstUncond)

numFeatures = 10

step = 0.1

gridRatioStdTrueToStdRawFcst = np.linspace(0.0, 1.0, 1.0 / step + 1)[1:-1]
nGrid = gridRatioStdTrueToStdRawFcst.size

SBD = dict()

for isBayes in [False, True]:
    SBD[isBayes] = dict()
    
    for name in NAMES_OF_POLICIES:
        SBD[isBayes][name] = dict()
        
        for key in LIST_OF_METRICS:
            SBD[isBayes][name]["mean" + key] = np.zeros(nGrid, dtype=np.double)
            SBD[isBayes][name]["stdErr" + key] = np.zeros(nGrid, dtype=np.double)
            
        #| for key in LIST_OF_METRICS:
    #| for name in NAMES_OF_POLICIES:
#| for isBayes in [False, True]:

seed = 123
size = 1000000


for iRatio, ratio in enumerate(gridRatioStdTrueToStdRawFcst):
    stdAlphaPrior = np.sqrt(variaRawFcstUncond * ratio)
    variaRawFcst = variaRawFcstUncond * (1.0 - ratio)
    
    shapePrecRawFcst, scalePrecRawFcst = \
            calibGammaFcstPrecision(variaRawFcst, numFeatures=numFeatures)
    
    for isBayes in [False, True]:
        for name in NAMES_OF_POLICIES:
            sizingPolicy = getSizingPolicy(name)
            res = summarizeGaussianPriorGammaFcstPrecisionExperiments(seed, size \
                    , meanAlphaPrior, stdAlphaPrior \
                    , shapePrecRawFcst, scalePrecRawFcst \
                    , constNoiseRet=constNoiseRet \
                    , sizingPolicy=sizingPolicy \
                    , applyBayesianShrinkage=isBayes \
                    )
            
            for key in LIST_OF_METRICS:
                SBD[isBayes][name]["mean" + key][iRatio] = res["mean" + key]
                SBD[isBayes][name]["stdErr" + key][iRatio] = res["stdErr" + key]
            
        print '.',

        #| for name in NAMES_OF_POLICIES:
    #| for isBayes in [False, True]:

for isBayes in [False, True]:
    for name in NAMES_OF_POLICIES:
        for key in LIST_OF_METRICS:
            if key == "BetQty":
                continue
            
            SBD[isBayes][name]["perQty" + key] = SBD[isBayes][name]["mean" + key] / SBD[isBayes][name]["meanBetQty"]


def nan_plot(plotter, x, y, *args, **kwargs):
    isNotNan = ~np.isnan(y)
    plotter.plot(x[isNotNan], y[isNotNan], *args, **kwargs)

def plotPerQtyPerformances(plotter, SBD, isBayes, name="binary", color="r"):
    nan_plot(plotter, gridRatioStdTrueToStdRawFcst, SBD[isBayes][name]["perQtyExpectedPnl"], "-", color=color)
    nan_plot(plotter, gridRatioStdTrueToStdRawFcst, SBD[isBayes][name]["perQtyModFcstPnl"], "--", color=color)
    nan_plot(plotter, gridRatioStdTrueToStdRawFcst, np.sqrt(SBD[isBayes][name]["perQtyRetTotalVaria"]) / constNoiseRet, ".", color=color)
    

if True:
    fig = plt.subplot(1, 1, 1)
    
    name = "hockey"
    
    plotPerQtyPerformances(fig, SBD, isBayes=False, name=name, color="g")
    plotPerQtyPerformances(fig, SBD, isBayes=True,  name=name, color="r")
    
    fig.grid()
    plt.show()

pass

