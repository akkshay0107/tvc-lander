from gym import PyEnvironment


def main():
    env = PyEnvironment(10000)
    # Test episode
    obs = env.reset()
    done = False
    step = 1
    tot_rew = 0
    while not done:
        action = [0, 0]
        obs, reward, done, _ = env.step(action)
        thrust = action
        print(f"{step=} {obs=} {thrust=} {reward=}")
        tot_rew += reward
        step += 1

    print("Test episode completed.")
    print(f"Total Reward: {tot_rew}")


if __name__ == "__main__":
    main()
