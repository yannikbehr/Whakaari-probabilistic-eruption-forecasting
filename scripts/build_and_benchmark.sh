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


# clean up playback files
function run_benchmark(){
    DIRECTORY=$1
    CLEANALL=$2
    WORKFLOWDIR=/home/bayes/src/whakaaribn/data/workflow
    if [ "${CLEANALL}" == "true" ]; then
        CMD="whakaaribn benchmark --directory /home/bayes/data --clean"
    else
        CMD="whakaaribn benchmark --directory /home/bayes/data --cores 5 "
    fi

    echo $CMD
    docker run --rm \
        -u $(id -u):$(id -g) \
        --entrypoint /bin/bash \
        -v $DIRECTORY:/home/bayes/data \
        ${IMAGE} -c "${CMD}"
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
    --cleanall              Clean raw data and input files before
                            running a new playback.
    --data                  Provide an alternative data root directory.
                            (Default: ${DATADIR})
EOF
}

# Processing command line options
while [ $# -gt 0 ]
do
    case "$1" in
        -b | --build) BUILD=true;;
        -i | --interactive) INTERACTIVE=true;;
        --print) PRINTEVENTS=true;;
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
    -f "${PROJECT_ROOT}/docker/Dockerfile" .
fi

if [ "${INTERACTIVE}" == "true" ]; then
    docker run -it --rm \
        -u $(id -u):$(id -g) \
        -v $DATADIR:/home/bayes/data \
        --entrypoint /bin/bash \
    ${IMAGE} 
    exit 0
fi

run_benchmark $DATADIR $CLEANALL
