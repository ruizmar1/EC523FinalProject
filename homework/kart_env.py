import gymnasium as gym
import numpy as np
import pystk
from utils import PyTux
import matplotlib.pyplot as plt
import cv2
from gymnasium.spaces import Box

RESCUE_TIMEOUT = 15

# custom SuperTuxKart gymnasium environment for this project's usage

class SuperTuxKartEnv(gym.Env):
    def __init__(self, track="lighthouse", max_frames=1000):
        super().__init__()
        # declaring variables to be used later
        self.track_name = track
        self.max_frames = max_frames
        self.pytux = PyTux(screen_width=128, screen_height=96)
        self.mode = "human"
    
        # Gym action space, all continuous until we discretize the discrete values
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # observation is the image of the frame of the game
        self.observation_space = Box(
            low=0, high=255, shape=(96, 128, 3), dtype=np.uint8
        )
        


    # looking at reset usage in utils.py and just applying it to this function called reset
    # reminder that reset is called before every race starts, so just need to think about what conditions need to be met at race start
    def reset(self, seed=None, options=None):
        #super().reset(seed=seed)
        # create a plot figure and axis for rendering
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots()

        print("Resetting the environment...")
        if self.pytux.k is not None:
            self.pytux.k.stop()
            del self.pytux.k

        # self config that was used in utils.py
        self.config = pystk.RaceConfig()
        self.config.num_kart = 1  
        self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.config.track = self.track_name 

        # starting race, similar to how it is done in utils.py
        self.pytux.k = pystk.Race(self.config)
        self.pytux.k.start()
        self.pytux.k.step()

        # update the track object and check its length, same as in utils.py
        self.track = pystk.Track()
        self.track.update()

        # declaring useful variables and setting them to zero
        self.step_count = 0
        self.last_rescue = 0
        self.t = 0

        # return method for gymnasium, required to return object and dictionary
        # note to ssa, this is litttle different
        obs = np.transpose(np.array(self.pytux.k.render_data[0].image), (1, 2, 0)).copy()  # From (C, H, W) -> (H, W, C)
        return obs, {}  # Add a batch dimension

    # defining what step looks lik in gym, copying a lot from utils.py
    def step(self, action):
        """
        Step the environment using the provided action.
        """
    
        # Increment the step count
        self.step_count += 1
        self.t += 1
        # Remove batch dimension
        # Update the world state
        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart


        # note to marghe and ssa, this might be issue for DDPG
        action = np.squeeze(action)  # (1, 5) -> (5,)

        # Now map the array into a pystk.Action
        pystk_action = pystk.Action()
        pystk_action.steer = float(action[0])
        pystk_action.acceleration = float(action[1])
        pystk_action.brake = bool(action[2] > 0.5)
        pystk_action.nitro = bool(action[3] > 0.5)
        pystk_action.drift = bool(action[4] > 0.5)

        # Detect crash or timeout
        if (np.linalg.norm(kart.velocity)) < 1 and self.t-self.last_rescue > RESCUE_TIMEOUT:
            print("Kart crashed! Respawning...")
            pystk_action.rescue = True
            self.last_rescue = self.t
        else:
            pystk_action.rescue = False

        # Step the environment using the pystk.Action object
        self.pytux.k.step([pystk_action])  # Correctly call self.pytux.k.step()

        # Detect if the car is flipped or stuck
        if (np.linalg.norm(kart.velocity) < 0.1 or abs(kart.rotation[2]) > 0.8) and self.t - self.last_rescue > RESCUE_TIMEOUT:
            print("Car is flipped or stuck! Rescuing...")
            pystk_action.rescue = True  # Rescue the car
            self.last_rescue = self.t

        # Update track
        self.track.update()
        track_length = self.track.length if self.track.length > 0 else 1.0


        # Get the new observation
        # transposed this too, might be good for DDPG
        obs = np.transpose(np.array(self.pytux.k.render_data[0].image), (1, 2, 0)).copy()  # From (C, H, W) -> (H, W, C)

        # Calculate the reward (you can modify this for more complex reward shaping)
        #reward = kart.overall_distance/1000
        progress = kart.overall_distance - getattr(self, "last_distance", 0)
        self.last_distance = kart.overall_distance

        # Use progress as the reward
        reward = progress

        # Add a penalty for being stuck
        if np.linalg.norm(kart.velocity) < 0.1:
            reward -= 0.1

        # Check if the episode is terminated or truncated
        terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
        truncated = self.step_count >= self.max_frames


        return obs, reward, terminated, truncated, {}

    # rendering image to see kart
    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        if self.mode == 'human':
            
            # getting image of the current track (using utils.py version)
            img = np.array(self.pytux.k.render_data[0].image)

            # clear the previous plot and show image
            self.ax.clear()
            self.ax.imshow(img)

            # adding race car current point, taking out for now because we dont need
            #WH2 = np.array([128, 96]) / 2
            #ax.add_artist(plt.Circle(WH2 * (1 + self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))


            # draw and then pause
            plt.draw()
            plt.pause(1e-3)
        
            # Close the figure to prevent memory overload (useful in a loop)
            if None:
                plt.close(self.fig)

            return img  # Or you can return other relevant information if need

    def close(self):
        self.pytux.close()

