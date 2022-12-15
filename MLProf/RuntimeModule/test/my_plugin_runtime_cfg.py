# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", "file:///afs/cern.ch/user/n/nprouvos/public/testfile.root")  # noqa
options.register("graphPath",
                "/afs/cern.ch/user/n/nprouvos/public/testfile.root",
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "Absolute path to the graph of the network to be used",
                 )
options.register("inputTensorName",
                "input",
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "Tensorflow name of the input into the given network",
                 )
options.register("outputTensorName",
                "Identity",
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "Tensorflow name of the output of the given network",
                 )
options.register("outputPath",
                "/afs/cern.ch/user/n/nprouvos/public/results.csv",  # just some existing path, not to be used
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "The path to save the csv/sql3 file with the results",
                 )

options.register("inputSize",
                10,
                VarParsing.multiplicity.singleton,
                VarParsing.varType.int,
                "Size of the input layer",
                 )
options.register("outputSize",
                1,
                VarParsing.multiplicity.singleton,
                VarParsing.varType.int,
                "Size of the output layer",
                 )
options.register("numberRuns",
                500,
                VarParsing.multiplicity.singleton,
                VarParsing.varType.int,
                "Number of events to be averaged upon for the time measurement",
                 )
options.register("numberWarmUps",
                100,
                VarParsing.multiplicity.singleton,
                VarParsing.varType.int,
                "Number of events to be tested before the begin of the time measurement",
                 )
options.register("batchsizes",
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                VarParsing.multiplicity.list,
                VarParsing.varType.int,
                "Different batchsizes to be tested",
                 )
options.parseArguments()

data = options.graphPath


# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles),
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# setup MyPluginRuntime by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MLProf.RuntimeModule.myPluginRuntime_cfi")
process.myPluginRuntime.graphPath = cms.string(data)
process.myPluginRuntime.inputTensorName = cms.string(options.inputTensorName)
process.myPluginRuntime.outputTensorName = cms.string(options.outputTensorName)
process.myPluginRuntime.filenameOutputCsv = cms.string(options.outputPath)
# add untracked?
process.myPluginRuntime.inputSize = cms.int32(options.inputSize)
process.myPluginRuntime.outputSize = cms.int32(options.outputSize)
process.myPluginRuntime.numberRuns = cms.int32(options.numberRuns)
process.myPluginRuntime.numberWarmUps = cms.int32(options.numberWarmUps)
# print(options.batchsizes, type(options.batchsizes), options.batchsizes[0], type(options.batchsizes[0]))
if type(options.batchsizes[0]) == type(options.batchsizes):
    process.myPluginRuntime.batchsizes = cms.vint32(options.batchsizes[0])
else:
    process.myPluginRuntime.batchsizes = cms.vint32(options.batchsizes)

# define what to run in the path
process.p = cms.Path(process.myPluginRuntime)
