# Multiple Bandit Problem solving using Optimistic Initial Values Algorithm

"""
This code is collected from the Udemy Tutorial
You can see the course from here
# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
"""
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m, upper_limit):  # Initialize each Bandit
        self.m = m  # True Mean
        self.mean = upper_limit  # Calculate the playout performance for each Bandit, our estimate of mean
        self.N = 1  # Total number of Turn Counter

    def pull(self):
        return np.random.randn() + self.m  # Pulling the Bandit Arm

    def update(self, x):  # Reward for the Bandits
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean + (1.0/self.N)*x


# Inside the main class
def run_experiment(m1, m2, m3, N, upper_limit=10):
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]

    data = np.empty(N)

    for i in range(N):
        # Optimistic Initial uses always greedy approach

        chosenBandit = np.argmax([b.mean for b in bandits])

        x = bandits[chosenBandit].pull()
        bandits[chosenBandit].update(x)

        # For the plot
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average


if __name__ == '__main__':
    bandit1_true_mean = 3.0
    bandit2_true_mean = 2.0
    bandit3_true_mean = 1.0
    play_turn_total = 100000
    c_1 = run_experiment(bandit1_true_mean, bandit2_true_mean, bandit3_true_mean, play_turn_total)

    # log scale plot
    plt.plot(c_1, label='Optimistic Initial')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='Optimistic Initial')
    plt.legend()
    plt.show()
