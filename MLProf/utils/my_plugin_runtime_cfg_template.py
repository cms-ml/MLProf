# coding: utf-8

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
options = VarParsing("python")
options.register("batchsizes",
                [1],    # default
                VarParsing.multiplicity.list,
                VarParsing.varType.int,
                "Batch size to be tested",
                 )
options.register("filename",
                "results.csv",  # default
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "The name of the file to save the csv file with the results",
                 )
options.parseArguments()

data = "GRAPH_PATH_PLACEHOLDER"


# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring("file://" + "INPUT_FILES_PLACEHOLDER"),
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
)

# setup MyPluginRuntime by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MLProf.RuntimeModule.myPluginRuntime_cfi")
process.myPluginRuntime.graphPath = cms.string(data)
process.myPluginRuntime.inputTensorNames = cms.vstring(INPUT_TENSOR_NAME_PLACEHOLDER)
process.myPluginRuntime.outputTensorNames = cms.vstring(OUTPUT_TENSOR_NAME_PLACEHOLDER)
process.myPluginRuntime.filenameOutputCsv = cms.string("OUTPUT_DIRECTORY_PLACEHOLDER" + options.filename)
process.myPluginRuntime.inputType = cms.string("INPUT_TYPE_PLACEHOLDER")

process.myPluginRuntime.inputSizes = cms.vint32(INPUT_SIZE_PLACEHOLDER)
process.myPluginRuntime.inputLengths = cms.vint32(INPUT_CLASS_DIMENSION_PLACEHOLDER)
process.myPluginRuntime.numberRuns = cms.int32(NUMBER_RUNS_PLACEHOLDER)
process.myPluginRuntime.numberWarmUps = cms.int32(NUMBER_WARM_UPS_PLACEHOLDER)
#process.myPluginRuntime.batchsizes = cms.vint32(BATCH_SIZES_PLACEHOLDER)
process.myPluginRuntime.batchsizes = cms.vint32(list(options.batchsizes))
# process.myPluginRuntime.batchsizes = cms.int32(options.batchsizes)

# define what to run in the path
process.p = cms.Path(process.myPluginRuntime)
