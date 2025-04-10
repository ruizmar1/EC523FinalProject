from kart_env import SuperTuxKartEnv
import argparse

parser = argparse.ArgumentParser(description="Run SuperTuxKart on a selected track.")
parser.add_argument('--track', type=str, default='lighthouse', help='Name of the track to run')
args = parser.parse_args()

env = SuperTuxKartEnv(track=args.track)
obs, _ = env.reset()
t = 0
done = False
while not done:
    action = env.action_space.sample()  # Replace with your controller or RL agent
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    t += 1
    env.render(done)
print("Finished at t=", t)