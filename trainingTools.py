import numpy as np
import random
from collections import deque


class FlightController:

    def __init__(self, input_size, output_size, model, epsilon=1.0):
        """
        This is the object which handles the actual flying, training the NN, 
        and saving and loading the weight vectors. It takes the model you built
        in Keras as an input. If you want to use weights from a file
        make sure to set epsilon=0.0 in initialising.
        """
        # Input/output size
        self.input_size = input_size
        self.output_size = output_size

        # Learning parameters
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Storage for previous runs
        self.memory = deque(maxlen=10000)
        
        # Save the model
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        # Stores the previous state vectors, actions and rewards for training
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        This is the training function. You can probably ignore it, it
        simply follows the Q-learning algorithm previously discussed.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def __call__(self, state_vector):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_size)
        else:
            return np.argmax(self.model.predict(state_vector))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
