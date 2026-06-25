from PufferMO.pufferlib.ocean.resource_gathering.resource_gathering import test_performance
print("_____________Testing the implemented environment performance")
test_performance(timeout=10)
print("_____________Testing the original environment performance")
import time
import numpy as np
import mo_gymnasium as mo_gym

env = mo_gym.make("resource-gathering-v0")
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
print(f"MO-Gymnasium SPS: {steps / elapsed:.0f}")