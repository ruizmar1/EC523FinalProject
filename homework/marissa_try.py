from kart_env import SuperTuxKartEnv
import argparse
import torch #ADDED BY marghe might be wrong
import torch.nn as nn #ADDED BY marghe might be wrong
import numpy as np #ADDED BY marghe might be wrong
import cv2 #ADDED BY marghe might be wrong
from ddpg_continuous_action import Actor #ADDED BY marghe might be wrong
import gymnasium as gym

def preprocess_observation(obs):
    # If obs is already grayscale (2D), skip color conversion
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs  # Already grayscale

    # Resize to a smaller shape
    resized = cv2.resize(gray, (64, 64))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Flatten the image
    return normalized.flatten().astype(np.float32)

parser = argparse.ArgumentParser(description="Run SuperTuxKart on a selected track.")
parser.add_argument('--track', type=str, default='lighthouse', help='Name of the track to run') #Picks the track, if you want another track you can change this
args = parser.parse_args()

env = SuperTuxKartEnv(track=args.track)
obs, _ = env.reset()
obs = preprocess_observation(obs)
obs_dim = np.prod(preprocess_observation(obs).shape) #ADDED BY marghe might be wrong
# Load actor model

#--------------ADDED BY MARGHE MIGHT BE WRONG
# Mock env wrapper for model init
class DummyEnv:
    def __init__(self, obs_shape, action_space):
        self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_shape,), dtype=np.float32)
        self.single_action_space = action_space
        self.action_space = action_space

dummy_env = DummyEnv(obs_dim, env.action_space)

actor = Actor(dummy_env)
#actor.load_state_dict(torch.load("runs/<your_run_name>/ddpg_continuous_action.cleanrl_model")[0])
actor.eval()

#------------------------



# t = 0
# done = False
# while not done:
#     action = env.action_space.sample()  # Replace with your controller or RL agent
#     obs, reward, terminated, truncated, _ = env.step(action) 
#     obs = preprocess_observation(obs)
#     done = terminated or truncated
#     t += 1
#     env.render(done)
# print("Finished at t=", t)



#-------------------ADDED BY MARGHE MIGHT BE WRONG
### SSA TESTED UP TO HERE
## ok so what we need to do now is do the gradient descent update after the loop is done and then re run the loop, I think last time i did 50 iterations
done = False
t = 0
obs = preprocess_observation(obs)
while not done:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: [1, obs_dim]
        action = actor(obs_tensor).squeeze(0).numpy()  # shape: (5,)

    obs, reward, terminated, truncated, _ = env.step(action)
    obs = preprocess_observation(obs)
    done = terminated or truncated
    t += 1
    env.render(done)
print("Finished at t=", t)
#------------------------

#episode ends when the end of the track is reached or when it reaches 100000