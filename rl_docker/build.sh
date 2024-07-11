#!/bin/bash
set -e
set -u

PROJECTNAME=$(basename "$(cd "$(dirname "\$0")"; cd ..; pwd -P)" | tr '[:upper:]' '[:lower:]')

docker build --network host -t ${PROJECTNAME} -f Dockerfile .
