#!/bin/bash

# This script in executed in the $CMSSW_BASE/src directory after the initial "cmsenv" command and
# before "scram b" is called.

action() {
    # remove existing code
    rm -rf MLProf

    # copy the code from the repository
    cp -r "${MLP_BASE}/cmssw/MLProf" .

    # remove test files
    rm -rf MLProf/*/test

    # remove aot specific files
    rm -rf MLProf/*/plugins/aot
}
action "$@"

