import sys

import gym
import gym_chrome_dino  # This should be kept as it is registering the custom environments
import argparse
import pandas as pd
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from wandb.integration.sb3 import WandbCallback


"""
    Main method definition
"""

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--Observe", help="If used, no training is done, just playing", action='store_true')
parser.add_argument("-n", "--NoBrowser", help="run without UI", action='store_true')

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    def make_env():

        if args.Obstacle is None or int(args.Obstacle) == 1:
            if not args.NoBrowser:
                return Monitor(gym.make('ChromeDinoRLPo-v0'))
            else:
                return Monitor(gym.make('ChromeDinoRLPoNoBrowser-v0'))
        elif int(args.Obstacle) == 2:
            if not args.NoBrowser:
                return Monitor(gym.make('ChromeDinoRLPoTwoObstacles-v0'))
            else:
                return Monitor(gym.make('ChromeDinoRLPoTwoObstaclesNoBrowser-v0'))
        else:
            raise Exception('Just 1 or 2 obstacles are supported')

    OBSERVE = args.Observe

    if not OBSERVE:

        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 100000,
            "n_eval_episodes": 10,
            "env_name": "ChromeDinoRLPo-v0"
        }

        run = wandb.init(
            project='sb3',
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        num_cpu = 4
        env = SubprocVecEnv([lambda: make_env() for i in range(num_cpu)])
        # env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x%2000 == 0, video_length=200)
        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
        model.learn(
            total_timesteps=config["total_timesteps"],
            n_eval_episodes=config["n_eval_episodes"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2
            )
        )
        run.finish()

    else:

        score_df = pd.DataFrame(columns=['score'])

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        model = PPO.load(path="./best_models/dino_ppo", custom_objects=custom_objects)

        env = DummyVecEnv([lambda: make_env()])
        env.training = False
        obs = env.reset()

        for i in range(100):

            print(f'Rollout {i}')

            done = False

            while done is not True:

                try:

                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)

                    if done\
                            :
                        obs = env.reset()

                except Exception as e:
                    print('Closing environment due to exception')
                    env.close()
                    raise e

            score_df.loc[len(score_df)] = info['score']

        score_df.to_csv(path_or_buf='models/eval_ppo_wo_img.csv', index=False)
        env.close()
