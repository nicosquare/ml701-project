import gym
import gym_chrome_dino  # This should be kept as it is registering the custom environments
import argparse
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from wandb.integration.sb3 import WandbCallback


def make_env():
    return Monitor(gym.make(config["env_name"]))


"""
    Main method definition
"""

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--Observe", help="If used, no training is done, just playing", action='store_true')

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    OBSERVE = args.Observe

    if OBSERVE:

        # wandb.login(key='837737e201bd4bc505eca74b5406b0cb7a602db5')
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

        # TODO: Replace the route with the best model we got for Features PPO SB3
        model = PPO.load("./models/2od58sa7/model")

        env = DummyVecEnv([lambda: gym.make('ChromeDinoRLPo-v0')])
        env.training = False

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
