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
    "batchRules",
    [],
    VarParsing.multiplicity.list,
    VarParsing.varType.string,
    "Batch rules (format 'target_size:size_1,size_2,...') to be configured",
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
process.load("MLProf.RuntimeMeasurement.tfaotInference_cfi")
process.tfaotInference.outputFile = cms.string(options.csvFile)
process.tfaotInference.inputType = cms.string("__INPUT_TYPE__")
process.tfaotInference.batchRules = cms.vstring(options.batchRules)
process.tfaotInference.batchSize = cms.int32(options.batchSize)
process.tfaotInference.nCalls = cms.int32(__N_CALLS__)  # noqa

# define what to run in the path
process.p = cms.Path(process.tfaotInference)
