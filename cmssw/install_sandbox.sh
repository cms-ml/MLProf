#!/bin/bash

# This script in executed in the $CMSSW_BASE/src directory after the initial "cmsenv" command and
# before "scram b" is called.

rm -rf MLProf
cp -r "${MLP_BASE}/cmssw/MLProf" .
rm -rf MLProf/*/test
