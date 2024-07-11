#!/bin/bash
set -e
set -u

SCRIPTROOT="$( cd "$(dirname "\$0")" ; pwd -P )"
PROJECTNAME=$(basename "$(cd "$(dirname "\$0")"; cd ..; pwd -P)" | tr '[:upper:]' '[:lower:]')
HOMENAME=$(basename $HOME)

display_flag=""
gpus=""

usage() {
  echo "Usage: bash $0 -g <gpus> -d <true/false>" 1>&2
  exit 1
}

while getopts "g:d:" opt; do
  case $opt in
    g)
      gpus=$OPTARG
      ;;
    d)
      display_flag=$OPTARG
      ;;
    ?) 
      usage
      exit 1
      ;;
  esac
done

shift $(( $OPTIND - 1 ))

errors=""

if [ -z "$gpus" ]; then
  errors+="GPUs parameter is required.\n"
elif ! [[ "$gpus" =~ ^[1-9]+$ ]] && [ "$gpus" != "all" ]; then
  errors+="Invalid GPUs parameter. Must be a numeric value or 'all'.\n"
fi

if [ -z "${display_flag}" ]; then
  errors+="Display parameter is required.\n"
fi

if [ -n "$errors" ]; then
  echo -e "$errors" >&2
  usage
else
  echo -e "Use Docker Image: $PROJECTNAME"
  echo -e "Start Docker Container: ${PROJECTNAME}_container"
  echo -e "Use GPU: $gpus"
  echo -e "Use Display: $display_flag\n"
fi

if [ "$display_flag" = true ]; then
  export DISPLAY=$DISPLAY
  echo "Setting display to $DISPLAY"
  xhost +
  docker run -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/${HOMENAME}/.Xauthority:/root/.Xauthority \
    -e DISPLAY=$DISPLAY \
    -v ${SCRIPTROOT}/..:/home/root/rl_ws \
    --network=host \
    --gpus="${gpus}" \
    --name="${PROJECTNAME}_container" \
    ${PROJECTNAME} /home/root/rl_ws/rl_docker/setup.sh
else
  echo "Running Docker without display"
  docker run -it --rm \
    -v ${SCRIPTROOT}/..:/home/root/rl_ws \
    --network=host \
    --gpus="${gpus}" \
    --name="${PROJECTNAME}_container" \
    ${PROJECTNAME} /home/root/rl_ws/rl_docker/setup.sh
fi
