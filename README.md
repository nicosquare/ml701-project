<h1>Which Dino is The Best? 🦖</h1>
<p>
    <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
    <a href="https://apache.org/licenses/LICENSE-2.0.txt" target="_blank">
        <img alt="License: Apache License 2.0" src="https://img.shields.io/badge/License-Apache License 2.0-yellow.svg" />
    </a>
    <img alt="Python" src="https://img.shields.io/badge/python-v3.8-green" />
    <img alt="Conda" src="https://img.shields.io/badge/conda%7Cconda--forge-v3.7.1-important" />
</p>

<div style="text-align: justify">

Using AI algorithms to play games is a well-researched topic as it allows to test the algorithms in a well-defined environment handling multiple inputs and outputs simultaneously. This project implements and rigorously investigates genetic algorithms (GA) and reinforcement learning (RL) on the Chrome dinosaur (dino) game.

>For the genetic algorithm, we analyzed the effect  of changing the number of obstacles the dino sees ahead, the mutation rate, and the population size.

>In reinforcement learning, we utilized two approaches: deep Q-network (DQN) and proximal policy optimization (PPO), to train the dino. Finally, we compared in terms of their overall performance, training time, and limitations.

</div>

### ✨ [Click here to watch our dinos in action](https://shorturl.at/emoyW)

## Folder structure

>The root folder contains different main files for each one of the algorithms we tried:

### main_ga.py

<div style="text-align: justify">
    Launcher of our genetic algorithm, it accepts some parameters that allowed us
    to try different hyperparameters.
</div>

| Parameter            | Default  | Description                                    |
|:---------------------|:---------|:-----------------------------------------------|
| `-p`, `--Population` | `30`     | Number of individuals in a generation          |
| `-g`, `--Generation` | `10`     | Number of generations to run                   |
| `-m`, `--Mutation`   | `0.1`    | Mutation rate                                  |
| `-c`, `--Crossover`  | `0.8`    | Crossover rate                                 |
| `-obs`, `--Obstacle` | `1`, `2` | Number of obstacles to consider                |
| `-o`, `--Observe`    | `True`   | **If used, no training is done, just playing** |
| `-n`, `--NoBrowser`  | `False`  | Run without UI                                 |


```sh
  # Run with default values
  python main_ga.py
  # Defining parameters e.g. Population: 50, Generations: 10, MR: 0.5
  python main_ga.py -p 50 -g 10 -m 0-5
```

### main_rl_dqn.py

<div style="text-align: justify">
    Launcher of our DQN algorithm that uses images. There are
    two implementations here, one coded from scratch and one 
    using a <a target="_blank" href="https://github.com/DLR-RM/stable-baselines3/">stable-baselines-3</a> API.
</div>

| Parameter                  | Default  | Description                                    |
|:---------------------------|:---------|:-----------------------------------------------|
| `-i`, `--InitialEpsilon`   | `0.1`    | Initial epsilon                                |
| `-f`, `--FinalEpsilon`     | `0.0001` | Final epsilon                                  |
| `-s`, `--StepsToSave`      | `1000`   | Steps to save                                  |
| `-m`, `--MiniBatch`        | `16`     | **Mini batch size**                            |
| `-r`, `--Reward`           | `0.1`    | Game time reward                               |
| `-p`, `--Penalty`          | `-1.0`   | Game over penalty                              |
| `-l`, `--LearningRate`     | `1e-4`   | **Learning rate of the NN**                    |
| `-mid`, `--ModelId`        | `1`      | Id of model to use in the CNN                  |
| `-o`, `--Observe`          | `True`   | **If used, no training is done, just playing** |
| `-n`, `--NoBrowser`        | `False`  | Run without UI                                 |
| `-sb`, `--StableBaselines` | `False`  | Run Stable Baselines DQN                       |

Parameters in **bold** are available for Stable Baselines model (_with -sb flag_)

```sh
  # Run with default values
  python main_rl_dqn.py
  # Normal DQN defining parameters 
  python main_rl_dqn.py -i 1 -f 0.1
  # Stable-Baselines 3 DQN defining parameters 
  python main_rl_dqn.py -sb -m 32
```

### main_rl_dqn_wo_img.py

<div style="text-align: justify">
    Launcher of our DQN algorithm that uses features. There are
    two implementations here, one coded from scratch and one 
    using a <a target="_blank" href="https://github.com/DLR-RM/stable-baselines3/">stable-baselines-3</a> API.
</div>

| Parameter                  | Default  | Description                                    |
|:---------------------------|:---------|:-----------------------------------------------|
| `-i`, `--InitialEpsilon`   | `0.1`    | Initial epsilon                                |
| `-f`, `--FinalEpsilon`     | `0.0001` | Final epsilon                                  |
| `-s`, `--StepsToSave`      | `1000`   | Steps to save                                  |
| `-m`, `--MiniBatch`        | `16`     | **Mini batch size**                            |
| `-r`, `--Reward`           | `0.1`    | Game time reward                               |
| `-p`, `--Penalty`          | `-1.0`   | Game over penalty                              |
| `-l`, `--LearningRate`     | `1e-4`   | **Learning rate of the NN**                    |
| `-mid`, `--ModelId`        | `1`      | Id of model to use in the CNN                  |
| `-o`, `--Observe`          | `True`   | **If used, no training is done, just playing** |
| `-n`, `--NoBrowser`        | `False`  | Run without UI                                 |
| `-sb`, `--StableBaselines` | `False`  | Run Stable Baselines DQN                       |

Parameters in **bold** are available for Stable Baselines model (_with -sb flag_)

```sh
  # Run with default values
  python main_rl_dqn_wo_img.py
  # Normal DQN defining parameters 
  python main_rl_dqn_wo_img.py -i 1 -f 0.1
  # Stable-Baselines 3 DQN defining parameters 
  python main_rl_dqn_wo_img.py -sb -m 32
```

### main_rl_ppo_wo_img.py

<div style="text-align: justify">
    Launcher of our PPO algorithm that uses features. it only uses <a target="_blank" href="https://github.com/DLR-RM/stable-baselines3/">stable-baselines-3</a> API.
</div>

| Parameter                  | Default  | Description                                    |
|:---------------------------|:---------|:-----------------------------------------------|
| `-o`, `--Observe`          | `True`   | **If used, no training is done, just playing** |

Parameters in **bold** are available for Stable Baselines model (_with -sb flag_)

```sh
  # Run with default values
  python main_rl_ppo_wo_img.py
  # PPO defining parameters 
  python main_rl_ppo_wo_img.py -o
```

The files **main_rl_vpg** and **main_rl_wo_img_vpg** were some approaches that didn't succeed in the implementation in the current state of the project.

>The sub-folders contain resources used in the different mains:

| Folder          | Description                                                                             |
|:----------------|:----------------------------------------------------------------------------------------|
| ga              | Contains the base files to define individuals and populations.                          |
| network         | Definition of the neural networks we used in the project.                               |
| utils           | Utilities used in general in the code.                                                  |
| gym_chrome_dino | Sub-module folder, it contains another repo we created for our custom Gym environments. |

## How to configure the environment

### With Conda (recommended)

<div style="text-align: justify">
    1. First start by creating a new Python environment using the provided
    configuration file <b>dino_env.yml</b>:
</div>

```sh
conda env create -f dino_env.yml
```
<div style="text-align: justify">
    2. Install submodule dependency <b>gym_chrome_dino</b>, this is a separate
    project in which we coded the Gym environments we needed. First start by
    fetching the repository. Go to root folder and execute:
</div>️

```sh
# Fetch the latest version of submodule
git submodule update --init --recursive
# Uninstall (in case it was installed) and Re-install the module with pip
cd gym_chrome_dino && pip uninstall -y gym-chrome-dino && pip install -e . && cd ..
```

<div style="text-align: justify">
    3. When the environment completes its configuration, just access the environment
    and launch one of the configurations mentioned in the previous section:
</div>

```sh
# Activate environment
conda activate dino_env
# e.g. Run a GA with Mutation Rate 0.9
python main_ga.py -m 0.9
```
### With other environment managers

<div style="text-align: justify">
    We provide a file <b>requirements.txt</b> to install all the dependencies
    with a package manager like <b><a target="_blank" href="https://pip.pypa.io/en/stable/cli/pip_install/">pip</a></b>.
</div>

```sh
# Install with pip
pip install -r requirements.txt
```

<div style="text-align: justify">
    When completed, just execute the same steps 2 and 3 from the <b>With Conda</b> subsection.
</div>

## Run the current best models

<div style="text-align: justify">
    In the folder <b>best_models</b> you can find the different models we were able to train.
    Currently, we support the following:
</div>

#### GA

```sh
python main_ga.py -o
```

#### PPO with Stable Baselines 3

```sh
python main_rl_ppo_wo_img.py -o
```

#### Images DQN with Stable Baselines 3

```sh
python main_rl_dqn.py -sb -o
```

#### Features DQN with Stable Baselines 3

```sh
python main_rl_dqn_wo_img.py -sb -o
```


## Common issues

#### When running I've been asked for a Wandb (W&B) account

<div style="text-align: justify">
    We use <a target="_blank" href="https://wandb.ai/">wandb</a> to plot our results, you can create an account to
    visualize the progress of an execution, or just chose the option 3, it
    will ignore any logging.
</div>

```sh
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

#### I got an error regarding Chrome version

<div style="text-align: justify">
    To run the Chrome dino game locally, we need to download the appropriate
    chromedriver file. We use a script that downloads it automatically, but it
    downloads the latest version. If your computer runs other version, you need
    to download the correct version from <a target="_blank" href="https://chromedriver.chromium.org/downloads">here</a>
    and paste it in the root of the project. 
</div>

```sh
selenium.common.exceptions.SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version 97
Current browser version is 96.0.4664.93 with binary path /usr/bin/google-chrome
```

<div style="text-align: justify">
In this example, the version <b>96.0.4664.93</b> should be downloaded and replaced.
</div>

## Authors

|              |                       🧑🏻 **Nicolás Cuadrado**                        |                       👩🏼 **Sarah AlBarri**                       |                    👨🏻 **Yu Kang Wong**                     |
|--------------|:----------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------:|
| **Github**   |              [@nicosquare](https://github.com/nicosquare)              |          [@SarahAlBarri](https://github.com/SarahAlBarri)          |         [@yukang1996](https://github.com/yukang1996)         |
| **LinkedIn** | [@nicolascuadrado](https:\/\/www.linkedin.com\/in\/nicolascuadrado\/)  | [@sarah-albarri](https:\/\/www.linkedin.com\/in\/sarah-albarri\/)  | [@wongyukang](https:\/\/www.linkedin.com\/in\/wongyukang\/)  |

## Acknowledgements

Thanks to the creators of the open source code that was the seed of this project.

 - [robertjankowski](https://github.com/robertjankowski/ga-openai-gym)
- [elvisyjlin](https://github.com/elvisyjlin/gym-chrome-dino)

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2021 [Nicolás Cuadrado, Sarah AlBarri, Yu Kang Wong].<br />
This project is [Apache License 2.0](https://apache.org/licenses/LICENSE-2.0.txt) licensed.

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_