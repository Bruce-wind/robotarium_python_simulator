import matplotlib.pyplot as plt
import numpy as np

def plot_trend():
    fig, ax = plt.subplots(1,2)
    rewards = np.loadtxt('sac_models/reward_20221021-165433.txt')
    ma_rewards = np.loadtxt('sac_models/ma_reward_20221021-165433.txt')

    ax[0].plot(rewards)
    ax[1].plot(ma_rewards)

    plt.show()

def draft():
    gamma = np.logspace(-4, 8)
    print(len(gamma))

if __name__ == '__main__':
    plot_trend()
    # draft()
