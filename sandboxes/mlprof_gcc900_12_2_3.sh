#!/usr/bin/env bash

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the generic CMSSW setup
    export MLP_SCRAM_ARCH="slc7_amd64_gcc900"
    export MLP_CMSSW_VERSION="CMSSW_12_2_3"
    export MLP_CMSSW_ENV_NAME="$( basename "${this_file%.sh}" )"
    export MLP_CMSSW_FLAG="1"  # increment when content changed

    # required function for defining a custom setup
    mlp_cmssw_setup() {
        # called in CMSSW_BASE/src
        ln -s "${MLP_BASE}/MLProf" .
    }

    source "${this_dir}/_setup_cmssw.sh" "$@"
}
action "$@"
