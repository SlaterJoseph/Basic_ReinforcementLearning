import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

batch_size = 32
n_episodes = 1001

# Stores the model
output_dir = './model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    """
    We want to sample random points of many games, so we don't rely heavily on one strategy
    IE if we start only going left we want to see the options from going right
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Memory where we store states

        self.gamma = 0.95  # Discount rate -- How much we value end outcomes vs current ones

        # The Agent can explore or exploit. Explore tries new things, while exploit optimizes what we know At the
        # start we want to explore as the agent doesn't know anything, but as time goes on we want to exploit more
        # However we always want to do some exploration, so we save a min value (From 100% to 1%)

        self.epsilon = 1.0  # Exploration rate -- We only explore at the beginning (no exploiting)
        self.epsilon_decay = 0.995  # Decay our epsilon so over time we do more exploit, less explore
        self.epsilon_min = 0.01

        self.learning_rate = 0.001  # The step size for our stochastic gradient descent

        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the network
        Input of variables from state
        2 hidden layers
        Output possible actions
        :return:
        """
        model = Sequential()

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # Input from our states
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Our action size is our possible outputs
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        :param state: State of the current timestamp
        :param action: Action of the current timestamp
        :param reward: Reward of the current timestamp
        :param next_state: Prediction of the next state
        :param done: Lets us know if the episode has ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        :param state:
        :return:
        """

        # Exploration: If our random is less than epsilon, explore
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)  # Exploit: Predict best action
        return np.argmax(act_values[0])  # returns the action which is the best choice

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)  # create minibatch from memory

        for state, action, reward, next_state, done in minibatch:
            # We have ended the game (max time or dying)
            target = reward  # We finished, we know the reward (0 or 1)

            if not done:  # game was not completed
                # target is the base reward, plus discount (discounts future award)
                # Times the estimate of future award (which we get from the neural network)
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target  # Maps our action to the reward

            # X-axis is the current state, Y-axis is the predicted future reward
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:  # If we still explore often, decay it
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size)

done = False
for e in range(n_episodes):
    state = env.reset()  # reset the state
    state = np.reshape(state, [1, state_size])

    for time in range(5000):  # If the game lasts 5000 time steps
        # env.render()

        action = agent.act(state)  # Get an action (0 - left, 1 - right)
        # Early this will be random, but as time passes it will be more logical (Exploiting)
        next_state, reward, done, truncated, _ = env.step(action)
        reward = reward if not done else -10  # Penalize poor actions
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:  # To see how the agent performs if completed
            print(f"Episode: {e}/{n_episodes}, score: {time}, e: {agent.epsilon:.2}")
            break

        if len(agent.memory) > batch_size:  # Gives a chance for the agent to update
            agent.replay(batch_size)

