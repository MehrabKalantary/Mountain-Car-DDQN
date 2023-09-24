import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from collections import deque


class DDQN:
    def __init__(self, env, gamma=0.03, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.005, tau=0.125):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=1)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


env = gym.make('MountainCar-v0')
gamma = 0.9
epsilon = 0.95
trails = 1000
trail_len = 500
ddqn_agnet = DDQN(env=env)

steps = []
for trail in range(trails):
    current_state = env.reset()[0].reshape(1, 2)
    for step in range(trail_len):
        print('#', step)
        action = ddqn_agnet.act(current_state)
        new_state, reward, done, _, _ = env.step(action)
        new_state = new_state.reshape(1, 2)
        ddqn_agnet.remember(current_state, action, reward, new_state, done)
        ddqn_agnet.replay()
        ddqn_agnet.target_train()
        current_state = new_state
        if done:
            break

    if step >= 199:
        print('Faild')
    else:
        print('Success')
        ddqn_agnet.save_model('DDQN Model')
        break
