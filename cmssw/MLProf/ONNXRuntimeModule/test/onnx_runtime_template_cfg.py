# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
options = VarParsing("python")
options.register(
    "batchSizes",
    [1],
    VarParsing.multiplicity.list,
    VarParsing.varType.int,
    "Batch sizes to be tested",
)
options.register(
    "csvFile",
    "results.csv",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "The path of the csv file to save results",
)
options.parseArguments()


# define the process to run
process = cms.Process("MLPROF")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(__N_EVENTS__),  # noqa
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(*__INPUT_FILES__),  # noqa
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# setup options for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)


# setup MyPlugin by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MLProf.ONNXRuntimeModule.onnxRuntimePlugin_cfi")
process.onnxRuntimePlugin.graphPath = cms.string("__GRAPH_PATH__")
process.onnxRuntimePlugin.inputTensorNames = cms.vstring(__INPUT_TENSOR_NAMES__)  # noqa
process.onnxRuntimePlugin.outputTensorNames = cms.vstring(__OUTPUT_TENSOR_NAMES__)  # noqa
process.onnxRuntimePlugin.outputFile = cms.string(options.csvFile)
process.onnxRuntimePlugin.inputType = cms.string("__INPUT_TYPE__")
process.onnxRuntimePlugin.inputRanks = cms.vint32(__INPUT_RANKS__)  # noqa
process.onnxRuntimePlugin.flatInputSizes = cms.vint32(__FLAT_INPUT_SIZES__)  # noqa
process.onnxRuntimePlugin.batchSize = cms.int32(options.batchSizes[0])
process.onnxRuntimePlugin.nCalls = cms.int32(__N_CALLS__)  # noqa

# define what to run in the path
process.p = cms.Path(process.onnxRuntimePlugin)
