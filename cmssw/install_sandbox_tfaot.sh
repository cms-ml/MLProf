#!/bin/bash

# This script in executed in the $CMSSW_BASE/src directory after the initial "cmsenv" command and
# before "scram b" is called.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # check the aot config
    local aot_config="$1"
    if [ ! -f "${aot_config}" ]; then
        >&2 echo "aot config file not found: ${aot_config}"
        return "1"
    fi

    # define additional model variables
    local tool_name="tfaot-model-mlprof-test"

    # temporarily prepend the local cms_tfaot path
    export PYTHONPATH="${MLP_BASE}/modules/cms-tfaot:${PYTHONPATH}"

    # remove existing code
    rm -rf MLProf

    # copy the code from the repository
    cp -r "${MLP_BASE}/cmssw/MLProf" .

    # remove test files
    rm -rf MLProf/*/test

    # remove non-aot specific files
    rm -rf MLProf/*/plugins/*.{xml,cc}

    # compile the model
    local aot_dir="${CMSSW_BASE}/mlprof_aot"
    rm -rf "${aot_dir}"
    mkdir -p "${aot_dir}"
    cms_tfaot_compile \
        -c "${aot_config}" \
        -o "${aot_dir}" \
        --tool-name "${tool_name}" \
        --dev \
    || return "$?"

    # setup the tool
    scram setup "${aot_dir}/${tool_name}.xml" || return "$?"

    # move aot files
    for d in MLProf/*/plugins; do
        [ -d "${d}/aot" ] && mv "${d}/aot"/* "${d}"
        rm -rf "${d}/aot"
    done

    # fill template variables *.cc files
    local header_file="$( ls -1 "${aot_dir}/include/${tool_name}"/*_bs*.h | head -n 1 )"
    for f in MLProf/*/plugins/*.cc; do
        python3 "${this_dir}/render_aot.py" "${f}" "${header_file}" || return "$?"
    done
}
action "$@"
