#!/bin/bash
set -e
set -u

############################
# ADD YOUR OWN REQUIREMENT #
############################
pip install -e ./isaacgym/python
pip install -e ./rsl_rl
pip install -e ./legged_gym

/bin/bash