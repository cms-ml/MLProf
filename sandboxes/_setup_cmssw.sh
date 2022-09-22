#!/usr/bin/env bash

# Script that installs, removes and / or sources a CMSSW environment.

setup_cmssw() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig_dir="$( pwd )"


    #
    # get and check arguments
    #

    local mode="${1:-}"
    if [ ! -z "${mode}" ] && [ "${mode}" != "clear" ] && [ "${mode}" != "reinstall" ] && [ "${mode}" != "install_only" ]; then
        >&2 echo "unknown CMSSW source mode '${mode}'"
        return "1"
    fi


    #
    # check required global variables
    #

    if [ "$( type -t mlp_cmssw_setup )" != "function" ]; then
        >&2 echo "mlp_cmssw_setup must is not set but required by ${this_file} to setup CMSSW"
        return "2"
    fi
    if [ -z "${MLP_SCRAM_ARCH}" ]; then
        >&2 echo "MLP_SCRAM_ARCH is not set but required by ${this_file} to setup CMSSW"
        return "3"
    fi
    if [ -z "${MLP_CMSSW_VERSION}" ]; then
        >&2 echo "MLP_CMSSW_VERSION is not set but required by ${this_file} to setup CMSSW"
        return "4"
    fi
    if [ -z "${MLP_CMSSW_BASE}" ]; then
        >&2 echo "MLP_CMSSW_BASE is not set but required by ${this_file} to setup CMSSW"
        return "5"
    fi
    if [ -z "${MLP_CMSSW_ENV_NAME}" ]; then
        >&2 echo "MLP_CMSSW_ENV_NAME is not set but required by ${this_file} to setup CMSSW"
        return "6"
    fi
    if [ -z "${MLP_CMSSW_FLAG}" ]; then
        >&2 echo "MLP_CMSSW_FLAG is not set but required by ${this_file} to setup CMSSW"
        return "7"
    fi


    #
    # start the setup
    #

    local install_base="${MLP_CMSSW_BASE}/${MLP_CMSSW_ENV_NAME}"
    local install_path="${install_base}/${MLP_CMSSW_VERSION}"
    local pending_flag_file="${MLP_CMSSW_BASE}/pending_${MLP_CMSSW_ENV_NAME}_${MLP_CMSSW_VERSION}"
    export MLP_SANDBOX_FLAG_FILE="${install_path}/mlp_flag"

    # ensure MLP_CMSSW_BASE exists
    mkdir -p "${MLP_CMSSW_BASE}"

    # remove the current installation
    if [ "${mode}" = "clear" ] || [ "${mode}" = "reinstall" ]; then
        echo "removing current installation at $install_path (mode '${mode}')"
        rm -rf "${install_path}"

        # optionally stop here
        [ "${mode}" = "clear" ] && return "0"
    fi

    # check if we need to wait for another install process to finish
    if [ ! -d "${install_path}" ]; then
        # from here onwards, files and directories could be created and in order to prevent race
        # conditions from multiple processes, guard the setup with the pending_flag_file and
        # sleep for a random amount of seconds between 0 and 10 to further reduce the chance of
        # simultaneously starting processes getting here at the same time
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
            # wait at most 10 minutes
            sleep_counter="$(( $sleep_counter + 1 ))"
            if [ "${sleep_counter}" -ge 120 ]; then
                >&2 echo "cmssw ${MLP_CMSSW_VERSION} is setup in different process, but number of sleeps exceeded"
                return "8"
            fi
            echo -e "\x1b[0;49;36mcmssw ${MLP_CMSSW_VERSION} already being setup in different process, sleep ${sleep_counter} / 120\x1b[0m"
            sleep 5
        done
    fi

    # install from scratch when not present
    if [ ! -d "${install_path}" ]; then
        local ret
        touch "${pending_flag_file}"
        echo "installing ${MLP_CMSSW_VERSION} in ${install_base}"

        (
            mkdir -p "${install_base}" || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )
            cd "${install_base}"
            source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )
            export SCRAM_ARCH="${MLP_SCRAM_ARCH}"
            scramv1 project CMSSW "${MLP_CMSSW_VERSION}" || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )
            cd "${MLP_CMSSW_VERSION}/src"

            # custom setup
            mlp_cmssw_setup || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )

            # compile
            eval "$( scramv1 runtime -sh )" || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )
            scram b || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )

            # write the flag into a file
            echo "version ${MLP_CMSSW_FLAG}" > "${MLP_SANDBOX_FLAG_FILE}"
            rm -f "${pending_flag_file}"
        ) || ( ret="$?" && rm -f "${pending_flag_file}" && return "${ret}" )
    fi

    # at this point, the src path must exist
    if [ ! -d "${install_path}/src" ]; then
        >&2 echo "src directory not found in CMSSW installation at ${install_path}"
        return "9"
    fi

    # check the flag and show a warning when there was an update
    if [ "$( cat "${MLP_SANDBOX_FLAG_FILE}" | grep -Po "version \K\d+.*" )" != "${MLP_CMSSW_FLAG}" ]; then
        >&2 echo ""
        >&2 echo "WARNING: the CMSSW software environment ${MLP_CMSSW_ENV_NAME} seems to be outdated"
        >&2 echo "WARNING: please consider removing (mode 'clear') or updating it (mode 'reinstall')"
        >&2 echo ""
    fi

    # optionally stop here
    [ "${mode}" = "install_only" ] && return "0"

    # source it
    source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" || return "$?"
    export SCRAM_ARCH="${MLP_SCRAM_ARCH}"
    export CMSSW_VERSION="${MLP_CMSSW_VERSION}"
    cd "${install_path}/src"
    eval "$( scramv1 runtime -sh )"
    cd "${orig_dir}"

    # prepend the presistent paths so that local packages are priotized
    export PATH="${MLP_PERSISTENT_PATH}:${PATH}"
    export PYTHONPATH="${MLP_PERSISTENT_PYTHONPATH}:${PYTHONPATH}"

    # mark this as a bash sandbox for law
    export LAW_SANDBOX="bash::\$MLP_BASE/sandboxes/${MLP_CMSSW_ENV_NAME}.sh"
}
setup_cmssw "$@"
