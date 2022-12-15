#!/bin/bash


cd "${CMSSW_BASE}/src"
scram b
echo "plugin compiled"
cmsRun MLProf/RuntimeModule/test/my_plugin_runtime_cfg.py ${@:1:9} ${@:11:${10}}
echo "time measurement done"

