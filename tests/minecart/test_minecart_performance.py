import time
import numpy as np

from pufferlib.ocean.minecart.minecart import Minecart, test_performance as port_perf


def test_port_performance():
    port_perf(timeout=10)


def test_reference_performance(timeout=10):
    import mo_gymnasium as mo_gym

    env = mo_gym.make("minecart-v0")
    env.reset(seed=0)
    steps = 0
    start = time.time()
    while time.time() - start < timeout:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
        steps += 1

    sps = steps / (time.time() - start)
    print(f"MO-Gymnasium Minecart SPS: {sps:.0f}")
    env.close()

if __name__ == "__main__":
    test_port_performance()
    print()
    test_reference_performance()
