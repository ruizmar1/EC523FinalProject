from kart_env import SuperTuxKartEnv

env = SuperTuxKartEnv(track='lighthouse')
obs, _ = env.reset()
t = 0
done = False
while not done:
    action = env.action_space.sample()  # Replace with your controller or RL agent
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    t += 1
    env.render()
print("Finished at t=", t)