# GetUp Gym

**Force-Guided Fall Recovery for Bipedal-Wheeled and Humanoid Robots**

This repository contains the open-source implementation of the paper:
> **Robust Fall Recovery for Armless Bipedal-Wheeled Robots via Force-Guided Learning**  
> Haidong Hou, Zhangguo Yu, Tao Han, Hengbo Qi, Ghazal Khaleel, Yu Zhang, Yidong Du, Xuechao Chen, and Fei Meng  
> *IEEE Robotics and Automation Letters*  
> project page at: https://2350575870.github.io/force-guided.github.io/

## Overview

We introduce **FTSR** (Force-guided Teacher-student framework with Stage-wise Rewards), which enables robust fall recovery for armless bipedal-wheeled robots and generalizes to high-DOF humanoid robots. The framework integrates:

- **Force-Guided Learning**: Height-correlated external auxiliary forces formulated as optimizable constraints within CPO.
- **Height-Progressive Stage-Wise Rewards**: Adaptive reward shaping that transitions from posture refinement to locomotion.
- **Teacher-Student Architecture**: Privileged information distillation for proprioceptive-only deployment.

## Supported Robots

- **Bipedal-Wheeled Robot** (`BipedalWheeled`): 8-DOF armless bipedal-wheeled robot (JiaRan).
- **Unitree Humanoid** (`UnitreeHumanoid`): 23-DOF humanoid robot for generalization validation.

## Installation

```bash
pip install -e .
```

Requires:
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) (Preview 4)
- PyTorch >= 1.13
- CUDA-capable GPU

## Training

```bash
# Bipedal-wheeled robot fall recovery
python scripts/train.py --task=BipedalWheeled --num_envs=4096 --headless

# Unitree humanoid fall recovery
python scripts/train.py --task=UnitreeHumanoid --num_envs=4096 --headless
```

## Evaluation

```bash
python scripts/play.py --task=BipedalWheeled --load_run=<run_name>
```

## Project Structure

```
getup_gym/
├── envs/
│   ├── bipedal_wheeled/     # Bipedal-wheeled robot environment
│   └── unitree_humanoid/    # Unitree humanoid environment
├── algorithms/              # PPO + Teacher-Student + Encoder MSE
├── modules/                 # Actor-Critic and Encoders
├── runners/                 # On-policy training runner
├── storage/                 # Rollout and encoder storage
├── common/                  # Base task, terrain, helpers, rewards
└── scripts/                 # train.py and play.py
```

## Citation

```bibtex
@article{hou2026force,
  title={Robust Fall Recovery for Armless Bipedal-Wheeled Robots via Force-Guided Learning},
  author={Hou, Haidong and Yu, Zhangguo and Han, Tao and Qi, Hengbo and Khaleel, Ghazal and Zhang, Yu and Du, Yidong and Chen, Xuechao and Meng, Fei},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```

## License

BSD-3-Clause
