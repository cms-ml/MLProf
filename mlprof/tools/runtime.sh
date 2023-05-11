#!/bin/bash


cd "${CMSSW_BASE}/src"
scram b && cmsRun MLProf/RuntimeModule/test/my_plugin_runtime_cfg.py "$@"
# ${1}
echo "time measurement done"

