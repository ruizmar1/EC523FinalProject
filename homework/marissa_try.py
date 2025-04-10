from kart_env import SuperTuxKartEnv

env = SuperTuxKartEnv(track='lighthouse')
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Replace with your controller or RL agent
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
print("done!")