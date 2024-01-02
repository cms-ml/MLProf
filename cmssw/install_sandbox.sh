#!/bin/bash

rm -rf MLProf
cp -r "${MLP_BASE}/cmssw/MLProf" .
rm -rf MLProf/*/test
