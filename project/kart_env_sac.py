import gymnasium as gym
import numpy as np
import pystk
from utils import PyTux
import matplotlib.pyplot as plt

RESCUE_TIMEOUT = 15


class SuperTuxKartEnv(gym.Env):
    def __init__(self, track_name, max_frames=1000):
        super().__init__()
        self.track_name = track_name
        self.max_frames = max_frames
        self.pytux = PyTux(screen_width=128, screen_height=96)
        self.mode = "human"
    
        self.action_space = gym.spaces.Dict({
            'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            'acceleration': gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            'brake': gym.spaces.Discrete(2),
            'nitro': gym.spaces.Discrete(2),
            'drift': gym.spaces.Discrete(2),
        })

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(96, 128, 3), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        self.fig, self.ax = plt.subplots()

        print("Resetting the environment...")
        if self.pytux.k is not None:
            self.pytux.k.stop()
            del self.pytux.k

        self.config = pystk.RaceConfig()
        self.config.num_kart = 1  
        self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.config.track = str(self.track_name)

        self.pytux.k = pystk.Race(self.config)
        self.pytux.k.start()
        self.pytux.k.step()

        self.track = pystk.Track()
        self.track.update()

        # Initialize all necessary variables
        self.step_count = 0
        self.last_rescue = 0
        self.t = 0
        self.crash_count = 0
        self.last_distance = 0.0  # Initialize progress tracking

        obs = np.array(self.pytux.k.render_data[0].image)
        info = {}
        info["final_observation"] = obs.copy()
        
        return obs, info

    def step(self, action_dict):
        self.step_count += 1
        self.t += 1
        
        terminated = self.t > 1000
        
        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart
        
        action = pystk.Action()
        action.steer = float(action_dict['steer'])
        action.acceleration = float(action_dict['acceleration'])
        action.brake = bool(action_dict['brake'])
        action.nitro = bool(action_dict['nitro'])
        action.drift = bool(action_dict['drift'])
        
        if (np.linalg.norm(kart.velocity) < 1.0) and (self.t - self.last_rescue > RESCUE_TIMEOUT):
            action.rescue = True
            self.last_rescue = self.t
            self.crash_count += 1
        
        self.pytux.k.step(action)
        self.track.update()
        track_length = max(self.track.length, 1.0)
        
        obs = np.array(self.pytux.k.render_data[0].image)
        
        # Reward calculation
        progress = kart.overall_distance - self.last_distance
        self.last_distance = kart.overall_distance
        
        velocity_reward = np.linalg.norm(kart.velocity) / 30.0
        
        dir_alignment = np.dot(
            kart.velocity[:2] / (np.linalg.norm(kart.velocity[:2]) + 1e-6),
            kart.front[:2] / (np.linalg.norm(kart.front[:2]) + 1e-6)
        )
        
        completion_bonus = 0.0
        if np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3):
            completion_bonus = 100.0
            terminated = True
        
        crash_penalty = -2.0 if action.rescue else 0.0
        
        reward = (
            5.0 * progress + 
            2.0 * velocity_reward + 
            1.5 * dir_alignment + 
            completion_bonus + 
            crash_penalty - 
            0.01  # time penalty
        )
        
        truncated = self.step_count >= self.max_frames
        
        info = {
            "progress": progress,
            "velocity": np.linalg.norm(kart.velocity),
            "crash_count": self.crash_count,
            "completion": kart.overall_distance / track_length,
            "final_observation": obs.copy() if (terminated or truncated) else None
        }
        
        return obs, np.array(reward), terminated, truncated, info

    def render(self, mode="human"):
        if self.mode == 'human':
            img = np.array(self.pytux.k.render_data[0].image)
            self.ax.clear()
            self.ax.imshow(img)
            plt.draw()
            plt.pause(1e-3)
            return img

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        self.pytux.close()
