# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
options = VarParsing("python")
options.setDefault("maxEvents", 1)
options.setDefault("maxEvents", 1)
options.register(
    "csvFile",
    "results.csv",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "path of the csv file to save results",
)
options.register(
    "batchSize",
    1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "batch size to be tested",
)
options.register(
    "inputType",
    "",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "input type; 'random', 'incremental', 'zeros', or 'ones'",
)
options.register(
    "nCalls",
    100,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "number of evaluation calls for averaging",
)
options.register(
    "batchRules",
    [],
    VarParsing.multiplicity.list,
    VarParsing.varType.string,
    "Batch rules (format 'target_size:size_1:size_2:...') to be configured",
)
options.parseArguments()


# define the process to run
process = cms.Process("MLPROF")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.maxEvents),
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles),
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# multi-threading options
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(1)
process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)

# setup the plugin
process.load("MLProf.RuntimeMeasurement.tfaotInference_cfi")
process.tfaotInference.outputFile = cms.string(options.csvFile)
process.tfaotInference.inputType = cms.string(options.inputType)
process.tfaotInference.batchRules = cms.vstring([r.replace(".", ",") for r in options.batchRules])
process.tfaotInference.batchSize = cms.int32(options.batchSize)
process.tfaotInference.nCalls = cms.int32(options.nCalls)

# define what to run in the path
process.p = cms.Path(process.tfaotInference)
