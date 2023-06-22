# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
options = VarParsing("python")
options.register(
    "batchSizes",
    [1],  # default
    VarParsing.multiplicity.list,
    VarParsing.varType.int,
    "Batch sizes to be tested",
)
options.register(
    "csvFile",
    "results.csv",  # default
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "The path of the csv file to save results",
)
options.parseArguments()


# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(NUMBER_EVENTS_TAKEN),
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(*INPUT_FILES_PLACEHOLDER),
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# setup MyPluginRuntime by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MLProf.RuntimeModule.myPluginRuntime_cfi")
process.myPluginRuntime.graphPath = cms.string("GRAPH_PATH_PLACEHOLDER")
process.myPluginRuntime.inputTensorNames = cms.vstring(INPUT_TENSOR_NAME_PLACEHOLDER)
process.myPluginRuntime.outputTensorNames = cms.vstring(OUTPUT_TENSOR_NAME_PLACEHOLDER)
process.myPluginRuntime.filenameOutputCsv = cms.string(options.csvFile)
process.myPluginRuntime.inputType = cms.string("INPUT_TYPE_PLACEHOLDER")

process.myPluginRuntime.inputSizes = cms.vint32(INPUT_SIZE_PLACEHOLDER)
process.myPluginRuntime.inputLengths = cms.vint32(INPUT_CLASS_DIMENSION_PLACEHOLDER)
process.myPluginRuntime.numberRuns = cms.int32(NUMBER_RUNS_PLACEHOLDER)
process.myPluginRuntime.numberWarmUps = cms.int32(NUMBER_WARM_UPS_PLACEHOLDER)
process.myPluginRuntime.batchSizes = cms.vint32(list(options.batchSizes))

# define what to run in the path
process.p = cms.Path(process.myPluginRuntime)
