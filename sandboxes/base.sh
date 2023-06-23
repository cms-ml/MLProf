#!/usr/bin/env bash

action() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the generic venv setup
    export MLP_VENV_NAME="$( basename "${this_file%.sh}" )"
    export MLP_VENV_REQUIREMENTS="${this_dir}/base.txt"

    source "${this_dir}/_setup_venv.sh" "$@"
}
action "$@"
