#!/usr/bin/env bash

setup_mlp() {
    # Runs the entire project setup, leading to a collection of environment variables starting with
    # "MLP_", the installation of the software stack via virtual environments.

    #
    # prepare local variables
    #

    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local micromamba_url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    local pyv="3.9"


    #
    # global variables
    # (MLP = MLProf)
    #

    # start exporting variables
    export MLP_BASE="${this_dir}"
    export MLP_DATA_BASE="${MLP_DATA_BASE:-${MLP_BASE}/data}"
    export MLP_SOFTWARE_BASE="${MLP_SOFTWARE_BASE:-${MLP_DATA_BASE}/software}"
    export MLP_CONDA_BASE="${MLP_CONDA_BASE:-${MLP_SOFTWARE_BASE}/conda}"
    export MLP_VENV_BASE="${MLP_VENV_BASE:-${MLP_SOFTWARE_BASE}/venvs}"
    export MLP_CMSSW_BASE="${MLP_CMSSW_BASE:-${MLP_SOFTWARE_BASE}/cmssw}"
    export MLP_STORE_LOCAL="${MLP_STORE_LOCAL:-${MLP_DATA_BASE}/store}"
    export MLP_LOCAL_SCHEDULER="${MLP_LOCAL_SCHEDULER:-true}"
    export MLP_SCHEDULER_HOST="${MLP_SCHEDULER_HOST:-127.0.0.1}"
    export MLP_SCHEDULER_PORT="${MLP_SCHEDULER_PORT:-8082}"

    # external variables
    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export X509_USER_PROXY="${X509_USER_PROXY:-/tmp/x509up_u$( id -u )}"
    export PYTHONWARNINGS="ignore"
    export GLOBUS_THREAD_MODEL="none"
    export VIRTUAL_ENV_DISABLE_PROMPT="${VIRTUAL_ENV_DISABLE_PROMPT:-1}"
    export MAMBA_ROOT_PREFIX="${MLP_CONDA_BASE}"
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"


    #
    # minimal local software setup
    #

    ulimit -s unlimited

    # persistent PATH and PYTHONPATH parts that should be
    # priotized over any additions made in sandboxes
    export MLP_PERSISTENT_PATH="${MLP_BASE}/bin:${MLP_BASE}/modules/law/bin:${MLP_SOFTWARE_BASE}/bin"
    export MLP_PERSISTENT_PYTHONPATH="${MLP_BASE}:${MLP_BASE}/modules/law"

    # prepend them
    export PATH="${MLP_PERSISTENT_PATH}:${PATH}"
    export PYTHONPATH="${MLP_PERSISTENT_PYTHONPATH}:${PYTHONPATH}"

    # remove parts of the software stack if requested
    if [ "${MLP_REINSTALL_CONDA}" = "1" ] || ( [ -z "${MLP_REINSTALL_CONDA}" ] && [ "${MLP_REINSTALL_SOFTWARE}" = "1" ] ); then
        echo "removing conda/micromamba at ${MLP_CONDA_BASE}"
        rm -rf "${MLP_CONDA_BASE}"
    fi
    if [ "${MLP_REINSTALL_VENV}" = "1" ] || ( [ -z "${MLP_REINSTALL_VENV}" ] && [ "${MLP_REINSTALL_SOFTWARE}" = "1" ] ); then
        echo "removing venvs at ${ML_VENV_BASE}"
        rm -rf "${ML_VENV_BASE}"
    fi
    if [ "${MLP_REINSTALL_CMSSW}" = "1" ] || ( [ -z "${MLP_REINSTALL_CMSSW}" ] && [ "${MLP_REINSTALL_SOFTWARE}" = "1" ] ); then
        echo "removing cmssw at ${ML_CMSSW_BASE}"
        rm -rf "${ML_CMSSW_BASE}"
    fi

    # conda base environment
    local conda_missing="$( [ -d "${MLP_CONDA_BASE}" ] && echo "false" || echo "true" )"
    if ${conda_missing}; then
        echo "installing conda/micromamba at ${MLP_CONDA_BASE}"
        (
            mkdir -p "${MLP_CONDA_BASE}"
            cd "${MLP_CONDA_BASE}"
            curl -Ls "${micromamba_url}" | tar -xvj -C . "bin/micromamba"
            ./bin/micromamba shell hook -y --prefix="${MLP_CONDA_BASE}" &> "micromamba.sh"
            mkdir -p "etc/profile.d"
            mv "micromamba.sh" "etc/profile.d"
            cat << EOF > ".mambarc"
changeps1: false
always_yes: true
channels:
  - conda-forge
EOF
        )
    fi

    # initialize conda
    source "${MLP_CONDA_BASE}/etc/profile.d/micromamba.sh" "" || return "$?"
    micromamba activate || return "$?"
    echo "initialized conda/micromamba"

    # install packages
    if ${conda_missing}; then
        echo
        echo "setting up conda/micromamba environment"

        # conda packages (nothing so far)
        micromamba install \
            libgcc \
            bash \
            "python=${pyv}" \
            git \
            git-lfs \
            || return "$?"
        micromamba clean --yes --all

        # update python base packages
        pip install --no-cache-dir -U pip setuptools wheel || return "$?"
    fi

    # source the base sandbox
    source "${MLP_BASE}/sandboxes/base.sh" "" || return "$?"

    # prepend persistent path fragments again to ensure priority for local packages
    export PATH="${MLP_PERSISTENT_PATH}:${PATH}"
    export PYTHONPATH="${MLP_PERSISTENT_PYTHONPATH}:${PYTHONPATH}"


    #
    # initialize / update submodules
    #

    for mpath in modules/law; do
        # do nothing when the path does not exist or it is not a submodule
        if [ ! -d "${mpath}" ] || [ ! -f "${mpath}/.git" ] ; then
            continue
        fi

        # initialize the submodule when the directory is empty
        if [ "$( ls -1q "${mpath}" | wc -l )" = "0" ]; then
            git submodule update --init --recursive "${mpath}"
        else
            # update when not on a working branch and there are no changes
            local detached_head="$( ( cd "${mpath}"; git symbolic-ref -q HEAD &> /dev/null ) && echo "true" || echo "false" )"
            local changed_files="$( cd "${mpath}"; git status --porcelain=v1 2> /dev/null | wc -l )"
            if ! ${detached_head} && [ "${changed_files}" = "0" ]; then
                git submodule update --init --recursive "${mpath}"
            fi
        fi
    done


    #
    # law setup
    #

    export LAW_HOME="${MLP_BASE}/.law"
    export LAW_CONFIG_FILE="${MLP_BASE}/law.cfg"

    if which law &> /dev/null; then
        # source law's bash completion scipt
        source "$( law completion )" ""

        # silently index
        law index -q
    fi

}

if setup_mlp "$@"; then
    echo -e "\x1b[0;49;35mMLProf successfully setup\x1b[0m"
    return "0"
else
    local code="$?"
    echo -e "\x1b[0;49;31mMLProf setup failed with code ${code}\x1b[0m"
    return "${code}"
fi
