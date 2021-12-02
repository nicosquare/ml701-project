import os

import gym
import gym_chrome_dino
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

# if __name__ == '__main__':
#     log_dir = './tmp'
#     stats_path = os.path.join(log_dir, 'vec_normalize.pkl')
#
#     # env = DummyVecEnv([lambda: gym.make('ChromeDinoRLPo-v0')])
#     # env = Monitor(env, log_dir)
#
#     os.makedirs(log_dir, exist_ok=True)
#
#     num_cpu = 1
#     env = SubprocVecEnv([lambda: gym.make('ChromeDinoRLPo-v0') for i in range(num_cpu)])
#     # env = VecFrameStack(env, n_stack=4)
#     # env = Monitor(env, log_dir)
#     env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10)
#
#     model = PPO('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=int(1))
#
#     model.save(log_dir+'/ppo_dino')
#
#     env.save(stats_path)
#
#
#
#     del model, env
#
#     env = DummyVecEnv([lambda:gym.make('ChromeDinoRLPo-v0')])
#     env = VecNormalize.load(stats_path, env)
#
#     env.training = False
#
#     env.norm_reward = False
#
#     model = PPO.load(log_dir+'/ppo_dino', env = env)
#
#     obs = env.reset()
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         # env.render()
def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)
    return env

if __name__ == '__main__':
    wandb.login(key='837737e201bd4bc505eca74b5406b0cb7a602db5')
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 125000,
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




