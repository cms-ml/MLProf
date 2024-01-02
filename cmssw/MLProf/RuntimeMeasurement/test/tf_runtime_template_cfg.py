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

# setup the plugin
process.load("MLProf.RuntimeMeasurement.tfRuntime_cfi")
process.tfRuntime.graphPath = cms.string("__GRAPH_PATH__")
process.tfRuntime.inputTensorNames = cms.vstring(__INPUT_TENSOR_NAMES__)  # noqa
process.tfRuntime.outputTensorNames = cms.vstring(__OUTPUT_TENSOR_NAMES__)  # noqa
process.tfRuntime.outputFile = cms.string(options.csvFile)
process.tfRuntime.inputType = cms.string("__INPUT_TYPE__")
process.tfRuntime.inputRanks = cms.vint32(__INPUT_RANKS__)  # noqa
process.tfRuntime.flatInputSizes = cms.vint32(__FLAT_INPUT_SIZES__)  # noqa
process.tfRuntime.batchSizes = cms.vint32(list(options.batchSizes))
process.tfRuntime.nCalls = cms.int32(__N_CALLS__)  # noqa

# define what to run in the path
process.p = cms.Path(process.tfRuntime)
