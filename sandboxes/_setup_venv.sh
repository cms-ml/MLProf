#!/usr/bin/env bash

# Script that installs and sources a virtual environment.

setup_venv() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"


    #
    # get and check arguments
    #

    local mode="${1:-}"


    #
    # check required global variables
    #

    if [ -z "${MLP_VENV_NAME}" ]; then
        >&2 echo "MLP_VENV_NAME is not set but required by ${this_file}"
        return "1"
    fi
    if [ -z "${MLP_VENV_REQUIREMENTS}" ]; then
        >&2 echo "MLP_VENV_REQUIREMENTS is not set but required by ${this_file}"
        return "2"
    fi

    # split $MLP_VENV_REQUIREMENTS into an array
    local requirement_files
    local requirement_files_contains_base="false"
    if ${shell_is_zsh}; then
        requirement_files=(${(@s:,:)MLP_VENV_REQUIREMENTS})
    else
        IFS="," read -r -a requirement_files <<< "${MLP_VENV_REQUIREMENTS}"
    fi
    for f in ${requirement_files[@]}; do
        if [ ! -f "${f}" ]; then
            >&2 echo "requirement file '${f}' does not exist"
            return "3"
        fi
        if [ "${f}" = "${MLP_BASE}/sandboxes/base.txt" ]; then
            requirement_files_contains_base="true"
        fi
    done
    local i0="$( ${shell_is_zsh} && echo "1" || echo "0" )"
    local first_requirement_file="${requirement_files[$i0]}"


    #
    # start the setup
    #

    local install_path="${MLP_VENV_BASE}/${MLP_VENV_NAME}"
    local venv_version="$( cat "${first_requirement_file}" | grep -Po "# version \K\d+.*" )"
    local pending_flag_file="${MLP_VENV_BASE}/pending_${MLP_VENV_NAME}"
    export MLP_SANDBOX_FLAG_FILE="${install_path}/mlp_flag"

    # the venv version must be set
    if [ -z "${venv_version}" ]; then
        >&2 echo "first requirement file ${first_requirement_file} does not contain a version line"
        return "4"
    fi

    # ensure the MLP_VENV_BASE exists
    mkdir -p "${MLP_VENV_BASE}"

    # remove the current installation
    if [ "${mode}" = "reinstall" ]; then
        echo "removing current installation at $install_path (mode '${mode}')"
        rm -rf "${install_path}"
    fi

    # from here onwards, files and directories could be created and in order to prevent race
    # conditions from multiple processes, guard the setup with the pending_flag_file and sleep for a
    # random amount of seconds between 0 and 10 to further reduce the chance of simultaneously
    # starting processes getting here at the same time
    if [ ! -f "${MLP_SANDBOX_FLAG_FILE}" ]; then
        local sleep_counter="0"
        sleep "$( python3 -c 'import random;print(random.random() * 10)')"
        # when the file is older than 30 minutes, consider it a dangling leftover from a
        # previously failed installation attempt and delete it.
        if [ -f "${pending_flag_file}" ]; then
            local flag_file_age="$(( $( date +%s ) - $( date +%s -r "${pending_flag_file}" )))"
            [ "${flag_file_age}" -ge "1800" ] && rm -f "${pending_flag_file}"
        fi
        # start the sleep loop
        while [ -f "${pending_flag_file}" ]; do
            # wait at most 15 minutes
            sleep_counter="$(( $sleep_counter + 1 ))"
            if [ "${sleep_counter}" -ge 180 ]; then
                >&2 echo "venv ${MLP_VENV_NAME} is setup in different process, but number of sleeps exceeded"
                return "5"
            fi
            echo -e "\x1b[0;49;36mvenv ${MLP_VENV_NAME} already being setup in different process, sleep ${sleep_counter} / 180\x1b[0m"
            sleep 5
        done
    fi

    # possible return value
    local ret="0"

    # install or fetch when not existing
    if [ ! -f "${MLP_SANDBOX_FLAG_FILE}" ]; then
        touch "${pending_flag_file}"
        echo "installing venv at ${install_path}"

        rm -rf "${install_path}"
        python3 -m venv --copies "${install_path}" || ( rm -f "${pending_flag_file}" && return "6" )

        # activate it
        source "${install_path}/bin/activate" "" || ( rm -f "${pending_flag_file}" && return "12" )

        # update pip
        echo -e "\n\x1b[0;49;35mupdating pip\x1b[0m"
        python3 -m pip install -U pip || ( rm -f "${pending_flag_file}" && return "13" )

        # install basoc production requirements
        if ! ${requirement_files_contains_base}; then
            echo -e "\n\x1b[0;49;35minstalling requirement file ${MLP_BASE}/sandboxes/base.txt\x1b[0m"
            python3 -m pip install -r "${MLP_BASE}/sandboxes/base.txt" || ( rm -f "${pending_flag_file}" && return "14" )
        fi

        # install requirement files
        for f in ${requirement_files[@]}; do
            echo -e "\n\x1b[0;49;35minstalling requirement file ${f}\x1b[0m"
            python3 -m pip install -r "${f}" || ( rm -f "${pending_flag_file}" && return "15" )
            echo
        done

        # write the version and a timestamp into the flag file
        echo "version ${venv_version}" > "${MLP_SANDBOX_FLAG_FILE}"
        echo "timestamp $( date "+%s" )" >> "${MLP_SANDBOX_FLAG_FILE}"
        rm -f "${pending_flag_file}"
    else
        # get the current version
        local curr_version="$( cat "${MLP_SANDBOX_FLAG_FILE}" | grep -Po "version \K\d+.*" )"
        if [ -z "${curr_version}" ]; then
            >&2 echo "the flag file ${MLP_SANDBOX_FLAG_FILE} does not contain a valid version"
            return "20"
        fi

        # complain when the version is outdated
        if [ "${curr_version}" != "${venv_version}" ]; then
            ret="21"
            >&2 echo ""
            >&2 echo "WARNING: outdated venv '${MLP_VENV_NAME}' located at"
            >&2 echo "WARNING: ${install_path}"
            >&2 echo "WARNING: please consider updating it by adding 'reinstall' to the source command"
            >&2 echo ""
        fi

        # activate it
        source "${install_path}/bin/activate" "" || return "$?"
    fi

    # export variables
    export MLP_VENV_NAME="${MLP_VENV_NAME}"

    # mark this as a bash sandbox for law
    export LAW_SANDBOX="bash::\$MLP_BASE/sandboxes/${MLP_VENV_NAME}.sh"

    return "${ret}"
}

setup_venv "$@"
