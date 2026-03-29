#!/usr/bin/env bash

##############################################################
# Build docker image for whakaaribn and run benchmarks       #
# 03/26 Y. Behr <y.behr@gns.cri.nz>                          #
##############################################################

# Make alias
shopt -s expand_aliases
set -euo pipefail

# Resolve the directory of this script (independent of where it is called from)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root = one level up from scripts/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
IMAGE=whakaaribn
DATADIR=/scratch/behrya/whakaaribn_benchmarks
BUILD=false
INTERACTIVE=false
CLEANALL=false
BUILDSTAGE=app
LOCAL=false
BACKEND=pgmpy
RUNTEST=false
DEV=false


# clean up playback files
function run_benchmark(){
    DIRECTORY=$1
    CLEANALL=$2
    BACKEND=$3
    WORKFLOWDIR=/home/bayes/src/whakaaribn/data/workflow
    if [ "${CLEANALL}" == "true" ]; then
        CMD="whakaaribn benchmark --directory /home/bayes/data --backend ${BACKEND} --clean"
    else
        CMD="whakaaribn benchmark --directory /home/bayes/data --backend ${BACKEND} --cores 5 "
    fi

    echo $CMD
    docker run --rm \
        -u $(id -u):$(id -g) \
        --entrypoint /bin/bash \
        -v $DIRECTORY:/home/bayes/data \
        ${IMAGE} -c "${CMD}"
}
  

function run_tests(){
    local dev_args=()
    local pip_prefix=""
    if [ "${DEV}" == "true" ]; then
        dev_args=(-v "${PROJECT_ROOT}:/home/bayes/src/whakaaribn")
        pip_prefix="pip install -q -e '/home/bayes/src/whakaaribn[dev]' && "
    fi
    docker run --rm \
        -u $(id -u):$(id -g) \
        --entrypoint /bin/bash \
        "${dev_args[@]+${dev_args[@]}}" \
        ${IMAGE} -c "${pip_prefix}hatch run test:run-pytest"
}


function usage(){
cat <<EOF
Usage: $0 [Options] 
Build and run docker for whakaaribn benchmarks.

Optional Arguments:
    -h, --help              Show this message.
    -b, --build             Rebuild the image.
    -i, --interactive       Start the container with a bash prompt.
    --image                 Provide alternative image name.
    --backend               BN backend to use: smile or pgmpy.
                            (Default: pgmpy)
    --cleanall              Clean raw data and input files before
                            running a new playback.
    --data                  Provide an alternative data root directory.
                            (Default: ${DATADIR})
    -t, --test              Run tests inside the container using
                            'hatch run test:run-pytest'.
    -d, --dev               Mount the project root into the container and
                            run 'pip install -e .[dev]' before executing.
                            Allows iterating without rebuilding the image.
                            Works with -t/--test and -i/--interactive.
EOF
}

# Processing command line options
while [ $# -gt 0 ]
do
    case "$1" in
        -b | --build) BUILD=true;;
        -i | --interactive) INTERACTIVE=true;;
        -t | --test) RUNTEST=true;;
        -d | --dev) DEV=true;;
        --print) PRINTEVENTS=true;;
        --backend) BACKEND="$2";shift;;
        --cleanall) CLEANALL=true;;
        --image) IMAGE="$2";shift;;
        --data) DATADIR="$2";shift;;
        -h) usage; exit 0;;
        -*) usage; exit 1;;
esac
shift
done


if [ "${BUILD}" == "true" ]; then
    echo "Building ${IMAGE}"
    DOCKER_BUILDKIT=1 docker build --target ${BUILDSTAGE} -t "${IMAGE}" \
    --build-arg D_UID=$(id -u) \
    --build-arg D_GID=$(id -g) \
    -f "${PROJECT_ROOT}/docker/Dockerfile" "${PROJECT_ROOT}"
fi

if [ "${RUNTEST}" == "true" ]; then
    run_tests
    exit 0
fi

if [ "${INTERACTIVE}" == "true" ]; then
    dev_vol_args=()
    if [ "${DEV}" == "true" ]; then
        dev_vol_args=(-v "${PROJECT_ROOT}:/home/bayes/src/whakaaribn")
    fi
    docker run -it --rm \
        -u $(id -u):$(id -g) \
        -v $DATADIR:/home/bayes/data \
        "${dev_vol_args[@]+${dev_vol_args[@]}}" \
        --entrypoint /bin/bash \
    ${IMAGE}
    exit 0
fi

mkdir -p "$DATADIR"
run_benchmark $DATADIR $CLEANALL $BACKEND
