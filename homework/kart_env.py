import gymnasium as gym
import numpy as np
import pystk
from utils import PyTux
import matplotlib.pyplot as plt

RESCUE_TIMEOUT = 15

class SuperTuxKartEnv(gym.Env):
    def __init__(self, track='lighthouse', max_frames=1000):
        super().__init__()
        self.track = track
        self.max_frames = max_frames
        self.pytux = PyTux(screen_width=128, screen_height=96)
        self.mode = "human"
    
        # Gym action space: steer [-1, 1], acceleration [0, 1], 3 binary flags
        self.action_space = gym.spaces.Dict({
            'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            'acceleration': gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            'brake': gym.spaces.Discrete(2),
            'nitro': gym.spaces.Discrete(2),
            'drift': gym.spaces.Discrete(2),
        })

        # Observation is the image (you could also include kart state info)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(96, 128, 3), dtype=np.uint8
        )

        self.step_count = 0
        self.last_rescue = 0

    def reset(self, seed=None, options=None):

        # Create a plot figure and axis for rendering
        self.fig, self.ax = plt.subplots()

        print("Resetting the environment...")
        if self.pytux.k is not None:
            self.pytux.k.stop()
            del self.pytux.k

        self.config = pystk.RaceConfig()
        self.config.num_kart = 1  # For a single kart in this case
        self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.config.track = 'lighthouse'  # Specify the track here


        self.pytux.k = pystk.Race(self.config)
        self.pytux.k.start()
        self.pytux.k.step()

        # Update the track object and check its length
        self.track = pystk.Track()
        self.track.update()

        self.step_count = 0
        self.last_rescue = 0
        self.t = 0

        obs = np.array(self.pytux.k.render_data[0].image)
        print(f"Observation shape: {obs.shape}")  # Add a print statement for debugging

        return obs, {}  # <- this part is critical!

    def step(self, action_dict):
        self.step_count += 1
        self.t += 1

        if self.t>1000:
            terminated = 1

        # Update world state
        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart

        # Build action
        action = pystk.Action()
        action.steer = float(action_dict['steer'])
        action.acceleration = float(action_dict['acceleration'])
        action.brake = bool(action_dict['brake'])
        action.nitro = bool(action_dict['nitro'])
        action.drift = bool(action_dict['drift'])

        # Detect crash or invalid kart state
        if (np.linalg.norm(kart.velocity)) < 1 and self.t-self.last_rescue> RESCUE_TIMEOUT:
            print("Kart crashed! Respawning...")
            action.rescue = True
            self.last_rescue = self.t

        # Step the simulation
        self.pytux.k.step(action)

        # Update track as a Track object (this should be done in reset as well)
        self.track = pystk.Track()
        self.track.update()
        # Get the track length safely
        track_length = self.track.length if self.track.length > 0 else 1.0  # avoid zero division


        # Get new observation
        obs = np.array(self.pytux.k.render_data[0].image)

        # Calculate reward (customize this as needed)
        reward = kart.overall_distance

        # Safety check for track length
        terminated = np.isclose(kart.overall_distance / track_length, 1.0, atol=2e-3)
        truncated = self.step_count >= self.max_frames

        return obs, reward, terminated, truncated, {}

    def respawn_kart(self, kart):
        """
        Respawn the kart at a close position to the crash site.
        You can customize the respawn location, but this is a basic example.
        """
        # Here we respawn the kart to a nearby location. You can adjust the offset as needed.
        crash_location = kart.location
        respawn_location = crash_location + np.array([5.0, 0.0, 0.0])  # Example respawn near the crash
        kart.location = respawn_location

        # Optionally, reset the kart's velocity, rotation, or any other states if necessary
        kart.velocity = np.array([0.0, 0.0, 0.0])  # Reset velocity
        kart.rotation = np.array([0.0, 0.0, 0.0])  # Reset rotation
        print(f"Respawning kart at location: {respawn_location}")


    def render(self):
        import matplotlib.pyplot as plt
        if self.mode == 'human':

            #state = pystk.WorldState()
            #state.update()
            #kart = state.players[0].kart
            
            # Assuming self.pytux.k.render_data[0].image is the image of the kart's view
            img = np.array(self.pytux.k.render_data[0].image)

            # Clear the previous plot
            self.ax.clear()

            # Show the current image
            self.ax.imshow(img)

            # Assuming you have projection and view matrices for the kart's location
            WH2 = np.array([128, 96]) / 2
            #ax.add_artist(plt.Circle(WH2 * (1 + self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))


            # Pause briefly to update the figure and create an animation effect
            plt.draw()
            plt.pause(1e-3)
        
            # Close the figure to prevent memory overload (useful in a loop)
            #plt.close(fig)

            return img  # Or you can return other relevant information if need

    def close(self):
        self.pytux.close()
