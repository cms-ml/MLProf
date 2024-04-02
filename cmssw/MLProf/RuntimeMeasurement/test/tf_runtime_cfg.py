# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
options = VarParsing("python")
options.register(
    "batchSize",
    1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Batch size to be tested",
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

# multi-threading options
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)

# setup the plugin
process.load("MLProf.RuntimeMeasurement.tfInference_cfi")
process.tfInference.graphPath = cms.string("__GRAPH_PATH__")
process.tfInference.inputTensorNames = cms.vstring(__INPUT_TENSOR_NAMES__)  # noqa
process.tfInference.outputTensorNames = cms.vstring(__OUTPUT_TENSOR_NAMES__)  # noqa
process.tfInference.outputFile = cms.string(options.csvFile)
process.tfInference.inputType = cms.string("__INPUT_TYPE__")
process.tfInference.inputRanks = cms.vint32(__INPUT_RANKS__)  # noqa
process.tfInference.flatInputSizes = cms.vint32(__FLAT_INPUT_SIZES__)  # noqa
process.tfInference.batchSize = cms.int32(options.batchSize)
process.tfInference.nCalls = cms.int32(__N_CALLS__)  # noqa

# define what to run in the path
process.p = cms.Path(process.tfInference)
