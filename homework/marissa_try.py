from kart_env import SuperTuxKartEnv
import argparse
def preprocess_observation(obs):
        # convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # resize to a smaller shape
        resized = cv2.resize(gray, (64, 64))
        # Normalize pixel values
        normalized = resized / 255.0
        # Flatten the image
        return normalized.flatten()

parser = argparse.ArgumentParser(description="Run SuperTuxKart on a selected track.")
parser.add_argument('--track', type=str, default='lighthouse', help='Name of the track to run')
args = parser.parse_args()

env = SuperTuxKartEnv(track=args.track)
obs, _ = env.reset()
obs = preprocess_observation(obs)
t = 0
done = False
while not done:
    action = env.action_space.sample()  # Replace with your controller or RL agent
    obs, reward, terminated, truncated, _ = env.step(action)
    obs = preprocess_observation(obs)
    done = terminated or truncated
    t += 1
    env.render(done)
print("Finished at t=", t)