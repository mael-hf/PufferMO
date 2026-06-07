from pufferlib.ocean.lunar_lander.Lunar_lander import test_performance
import time
import numpy as np
import mo_gymnasium as mo_gym
env = mo_gym.make("LunarLander-v2")
env.reset(seed=0)

steps = 0
start = time.time()
while time.time() - start < 10:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
    steps += 1

elapsed = time.time() - start
print("----Testing the implemented environment performance----")
test_performance(timeout=10)
print("----Testing the original environment performance----")
print(f"MO-Gymnasium SPS: {steps / elapsed:.0f}")