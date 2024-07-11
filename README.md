# rl_amp

This code aims to implement AMP on Legged_gym and rsl_rl with minimal modifications

**Code reference: [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)**

## Setup

Clone the code

```bash
git clone https://github.com/fan-ziqi/rl_amp.git
cd rl_amp
```

Download isaacgym

Build the docker image

```bash
cd rl_docker
bash build.sh
```

Run the docker container

```bash
bash run.sh -g <gpus, should be num 1~9 or all> -d <true/false>
# example: bash run.sh -g all -d true
```

## Usage

Train and play

```bash
python legged_gym/legged_gym/scripts/train.py --task=a1_amp --headless
python legged_gym/legged_gym/scripts/play.py --task=a1_amp
```

Retarget motions

```bash
python legged_gym/datasets/retarget_kp_motions.py
```
