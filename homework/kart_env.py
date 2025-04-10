import gymnasium as gym
import numpy as np
import pystk
from utils import PyTux
import matplotlib.pyplot as plt

RESCUE_TIMEOUT = 15

# custom SuperTuxKArt gymnasium environment for this project's usage

class SuperTuxKartEnv(gym.Env):
    def __init__(self, track, max_frames=1000):
        super().__init__()
        # declaring variables to be used later
        self.track = track
        self.max_frames = max_frames
        self.pytux = PyTux(screen_width=128, screen_height=96)
        self.mode = "human"
    
        # Gym action space: steer [-1, 1], acceleration [0, 1], 3 binary flags for brake, nitro, and drift
        self.action_space = gym.spaces.Dict({
            'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            'acceleration': gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            'brake': gym.spaces.Discrete(2),
            'nitro': gym.spaces.Discrete(2),
            'drift': gym.spaces.Discrete(2),
        })

        # observation is the image of the frame of the game
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(96, 128, 3), dtype=np.uint8
        )


    # looking at reset usage in utils.py and just applying it to this function called reset
    # reminder that reset is called before every race starts, so just need to think about what conditions need to be met at race start
    def reset(self, seed=None, options=None):

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
        self.config.track = self.track 


        # starting race, similar to how it is done in utils.py
        self.pytux.k = pystk.Race(self.config)
        self.pytux.k.start()
        self.pytux.k.step()

        # update the track object and check its length, same as un utils.py
        self.track = pystk.Track()
        self.track.update()

        # declaring useful variables and setting them to zero
        self.step_count = 0
        self.last_rescue = 0
        self.t = 0

        # return method for gymnasium, required to return object and dictionary, here our dictionary is blank since we dont want to add meta data rn
        obs = np.array(self.pytux.k.render_data[0].image)
        return obs, {}  

    # defining what step looks lik in gym, copying a lot from utils.py
    def step(self, action_dict):
        # updating steps and time step
        self.step_count += 1
        self.t += 1

        # IF the racer doesn't finish the track within 1000 time steps, just end game 
        if self.t>1000:
            terminated = 1

        # update world state at each step
        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart

        # building action
        action = pystk.Action()
        # adding action attributes, RIGHT NOW IT IS RANDOM in the future we will poll action from neural net
        action.steer = float(action_dict['steer'])
        action.acceleration = float(action_dict['acceleration'])
        action.brake = bool(action_dict['brake'])
        action.nitro = bool(action_dict['nitro'])
        action.drift = bool(action_dict['drift'])

        # detecting etect crash or  timeout
        if (np.linalg.norm(kart.velocity)) < 1 and self.t-self.last_rescue> RESCUE_TIMEOUT:
            print("Kart crashed! Respawning...")
            action.rescue = True
            self.last_rescue = self.t

        # step through the simulation
        self.pytux.k.step(action)

        # Update track
        self.track = pystk.Track()
        self.track.update()
        # get the track length 
        # adding stuff to avoid division
        track_length = self.track.length if self.track.length > 0 else 1.0  


        # new observation after stepping through the environment
        obs = np.array(self.pytux.k.render_data[0].image)

        # BASIC REWARD, WILL NEED TO DO SOME REWARD SHAPING LATER
        reward = kart.overall_distance

        # check for track length
        terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
        truncated = self.step_count >= self.max_frames

        return obs, reward, terminated, truncated, {}


    # rendering image to see kart
    def render(self, done):
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
            if done:
                plt.close(self.fig)

            return img  # Or you can return other relevant information if need

    def close(self):
        self.pytux.close()
