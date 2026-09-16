# Terraria Boss Beater via Deep Reinforcement Learning
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0.0-0B7285)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.6.0-4B32C3)](https://stable-baselines3.readthedocs.io/)
[![PPO](https://img.shields.io/badge/algorithm-PPO-E8733A)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/downloads/release/python-3110/)
[![tModLoader](https://img.shields.io/badge/tModLoader-C%23_mod-68217A)](https://github.com/tModLoader/tModLoader)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6)](#setup)
[![License](https://img.shields.io/badge/license-MIT-3DA639)](LICENSE)
### Introduction
This project utilizes proximal policy optimization(PPO) to train an agent to defeat Terraria bosses. The agent learns via screen capture and rewards based off of the situation, therefore possesses adaptivity. This allows training on any Terraria boss no matter if it is worm-type, segmented, or singular without any code changes. Thus, can be used with ease to explore possible play styles with different ingame equipment or for playtesting with different levels of competency against different bosses.

### Brief Overview on How it Works
![How it works diagram](https://github.com/jimmyjjz/TBB-DRL/blob/main/TBBDRL_diagram.png)

### The Gymnasium Environment
Terraria is wrapped as a standard [Gymnasium](https://gymnasium.farama.org/) environment in [tbb_env.py](DRL_component/tbb_env.py), so the game can be driven by anything that speaks the Gymnasium API. Training uses PPO from Stable-Baselines3, but swapping in A2C, SAC or another algorithm is a one line change.

```python
class TBBEnv(gym.Env):
    # left, right, down, jump/fly, attack, hook, heal, mouse x, mouse y
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    self.observation_space = spaces.Box(0, 255, shape=(HEIGHT // RESCALE_FACTOR, WIDTH // RESCALE_FACTOR, 3), dtype=np.uint8)
```

Observations are the screen grabbed with dxcam and downscaled by `RESCALE_FACTOR` (8 by default), so at 1920x1080 an observation is 135x240x3 uint8. [train_model.py](DRL_component/train_model.py) wraps the env in `VecFrameStack` with `n_stack=4`, so the policy sees four consecutive frames and can tell which way things are moving.

Actions are nine continuous values. The first seven are rounded to a key press or release, the last two are mapped to an absolute cursor position.

| Index | Input |
| --- | --- |
| 0 | move left (`a`) |
| 1 | move right (`d`) |
| 2 | move down (`s`) |
| 3 | jump / fly (`space`) |
| 4 | attack (left mouse) |
| 5 | hook (`e`) |
| 6 | heal (`h`) |
| 7 | cursor x |
| 8 | cursor y |

`reset()` presses `r` to reset the fight, `1` to select the weapon and `b` to use buff potions, moves the cursor to the middle of the screen, and releases every held key.

`step()` applies the action, grabs the next frame, then reads `reward_denoter.txt`. The reward is the change in that number since the last step. When the mod marks the fight inactive the episode ends, giving `win_reward` if the boss was defeated and `lose_reward` if the player died. Since the policy only ever sees pixels and a single number, nothing in the environment depends on which boss is being fought.

Note that all keys are modifiable, but I'd recommend using keys intuitive for playing the game.

### Setup
**Note: must have access to tModloader and Terraria**

First things first, Clone the repo.

Setup Gameside Training Mod:
1. Make a tModloader mod skeleton
2. Navigate to the folder of the mod skeleton you have created in the previous step and copy the contents of TBBTrainingMod_core into that folder(just copy paste and click replace files)
3. Replace "...\\settings.json" in line 8 of [SettingsAccesser](https://github.com/jimmyjjz/TBB-DRL/blob/main/TBBTrainingMod_core/SettingsAccesser.cs) with the path of [settings.json](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/settings.json)

Setup Deep Reinforcement Learning:
1. Replace "...\\reward_accumulant.txt" of [settings.json](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/settings.json) with the path of [reward_accumulant](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/reward_accumulant.txt).txt
2. [Install python 3.11](https://www.python.org/downloads/release/python-3110/) or [use conda(python built-in)](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) or [use pycharm(python built-in)(I am using this)](https://www.jetbrains.com/pycharm/download/?section=windows). More options exist.
3. Install dependencies. If pip is not installed(will be installed if using conda or pycharm or most if not all venv), [install it](https://pip.pypa.io/en/stable/installation/), and path via environmental variables.
```
pip install numpy==1.26.4 pillow==11.2.1 gymnasium==1.0.0 dxcam==0.0.5 pydirectinput==1.0.4 stable-baselines3==2.6.0
```

### How to Train and Run a Model
Run [train_model.py](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/run_model.py) to train. Run [run_model.py](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/run_model.py) to run. Make sure when running, an existing checkpoint is attempted to be used via [settings.json](https://github.com/jimmyjjz/TBB-DRL/blob/main/DRL_component/settings.json).

### Misc
While training and running works in whatever Terraria terrain, just in case you want a good orthrodox boss battle arena, one is provided [TBB-Environment World.wld](https://github.com/jimmyjjz/TBB-DRL/blob/main/TBB-Environment%20World.wld).

To play on this world just put [TBB-Environment World.wld](https://github.com/jimmyjjz/TBB-DRL/blob/main/TBB-Environment%20World.wld) into ...\\Terraria\\tModLoader\\Worlds
