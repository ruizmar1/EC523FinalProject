import gymnasium as gym
import numpy as np
import pystk
from utils import PyTux
import matplotlib.pyplot as plt
import cv2

RESCUE_TIMEOUT = 15

# custom SuperTuxKart gymnasium environment for this project's usage

class SuperTuxKartEnv(gym.Env):
    def __init__(self, track, max_frames=1000):
        super().__init__()
        # declaring variables to be used later
        self.track_name = track
        self.max_frames = max_frames
        self.pytux = PyTux(screen_width=128, screen_height=96)
        self.mode = "human"
    
        # Gym action space, all continuous until we discretize the discrete values
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), #steer, acc, break, drift, nitro (to check)
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # observation is the image of the frame of the game
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(96, 128, 3), dtype=np.uint8
        )
        


    # looking at reset usage in utils.py and just applying it to this function called reset
    # reminder that reset is called before every race starts, so just need to think about what conditions need to be met at race start
    def reset(self, seed=None, options=None):
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        # create a plot figure and axis for rendering
        self.fig, self.ax = plt.subplots()

        print("Resetting the environment...")
        if self.pytux.k is not None:
            self.pytux.k.stop()
            del self.pytux.k

        # self config that was used in utils.py
        self.config = pystk.RaceConfig()
        self.config.num_kart = 1  
        self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.config.track = self.track_name  # Use the track name string instead of Track object

        # starting race, similar to how it is done in utils.py
        self.pytux.k = pystk.Race(self.config)
        self.pytux.k.start()
        self.pytux.k.step()

        # Update track object - keep this separate from the config
        self.track = pystk.Track()
        self.track.update()

        # declaring useful variables and setting them to zero
        self.step_count = 0
        self.last_rescue = 0
        self.t = 0

        # return method for gymnasium
        obs = np.array(self.pytux.k.render_data[0].image)
        return obs, {}

    # def reset(self, seed=None, options=None):
    #     # Close any existing figure
    #     if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
    #         plt.close(self.fig)
        
    #     # Rest of your reset code...
    #     self.config = pystk.RaceConfig()
    #     self.config.num_kart = 1  
    #     self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    #     self.config.track = self.track_name
        
    #     # Initialize new race
    #     self.pytux.k = pystk.Race(self.config)
    #     self.pytux.k.start()
    #     self.pytux.k.step()
        
    #     # Reset tracking variables
    #     self.step_count = 0
    #     self.last_rescue = 0
    #     self.t = 0
        
    #     # Return initial observation
    #     obs = np.array(self.pytux.k.render_data[0].image)
    #     return obs, {}

    # defining what step looks lik in gym, copying a lot from utils.py
    #--------Marghe tried to change this to stop it from restarting randomly so I commented this step funciton out 
    #and added a new one

    # def step(self, action_array):
    #     # updating steps and time step
    #     self.step_count += 1
    #     self.t += 1

    #     # IF the racer doesn't finish the track within 1000 time steps, just end game 
    #     if self.t>1000:
    #         terminated = 1

    #     # update world state at each step
    #     state = pystk.WorldState()
    #     state.update()
    #     kart = state.players[0].kart

    #     # building action
    #     action = pystk.Action()
    #     # adding action attributes, RIGHT NOW IT IS RANDOM in the future we will poll action from neural net
    #     action.steer = float(action_array[0])
    #     action.acceleration = float(action_array[1])
    #     action.brake = bool(action_array[2] > 0.5)
    #     action.nitro = bool(action_array[3] > 0.5)
    #     action.drift = bool(action_array[4] > 0.5)

    #     # detecting etect crash or  timeout
    #     if (np.linalg.norm(kart.velocity)) < 1 and self.t-self.last_rescue> RESCUE_TIMEOUT:
    #         print("Kart crashed! Respawning...")
    #         action.rescue = True
    #         self.last_rescue = self.t

    #     # step through the simulation
    #     self.pytux.k.step(action)

    #     # Update track
    #     self.track = pystk.Track()
    #     self.track.update()
    #     # get the track length 
    #     # adding stuff to avoid division
    #     track_length = self.track.length if self.track.length > 0 else 1.0  


    #     # new observation after stepping through the environment
    #     obs = np.array(self.pytux.k.render_data[0].image)

    #     # BASIC REWARD, WILL NEED TO DO SOME REWARD SHAPING LATER
    #     reward = kart.overall_distance

    #     # check for track length
    #     terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
    #     truncated = self.step_count >= self.max_frames

    #     return obs, reward, terminated, truncated, {}

#MARGHE added the following to ty and fix the restarting envoroment issue
    def step(self, action_array):
        self.step_count += 1
        self.t += 1

        # Update world state
        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart

        # Build action
        action = pystk.Action()
        action.steer = float(action_array[0])
        action.acceleration = float(action_array[1])
        action.brake = bool(action_array[2] > 0.5)
        action.nitro = bool(action_array[3] > 0.5)
        action.drift = bool(action_array[4] > 0.5)

        # Detect crash or timeout
        if (np.linalg.norm(kart.velocity)) < 1 and self.t-self.last_rescue > RESCUE_TIMEOUT:
            print("Kart crashed! Respawning...")
            action.rescue = True
            self.last_rescue = self.t

        # Step through simulation
        self.pytux.k.step(action)

        # Update track
        self.track.update()
        track_length = self.track.length if self.track.length > 0 else 1.0

        # Get new observation
        obs = np.array(self.pytux.k.render_data[0].image)

        # Reward
        reward = kart.overall_distance

        # Termination conditions
        #terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
        #truncated = self.step_count >= self.max_frames or self.t >= 1000  # Combined time limit condition

        #Marghe trying to find the bug
        terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
        if terminated:
            print(f"Episode terminated after {self.step_count} steps (track completed!)")

        if self.step_count >= self.max_frames or self.t >= 1000:
            print(f"Episode truncated after {self.step_count} steps (time limit reached)")
            truncated = True
        else:
            truncated = False

        return obs, reward, terminated, truncated, {}

#--------------
    # rendering image to see kart
    def render(self, done=False):
        if self.mode == 'human':
            # Create figure if it doesn't exist
            if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                self.fig, self.ax = plt.subplots()
                plt.ion()  # Interactive mode on
                plt.show()
            
            # Get current game image
            img = np.array(self.pytux.k.render_data[0].image)
            
            # Update display
            self.ax.clear()
            self.ax.imshow(img)
            plt.draw()
            plt.pause(0.001)  # Small pause to allow GUI updates
            
            # Close figure if episode is done
            if done:
                plt.close(self.fig)
                delattr(self, 'fig')
                delattr(self, 'ax')
            
            return img

    def close(self):
        self.pytux.close()

