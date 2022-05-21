from datetime import datetime
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import time
import math
from DinoEnv import DinoEnv

# policy network
def OurModel(input_shape, action_space):

    input = tf.keras.layers.Input(input_shape)
    s = input

    c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu', activity_regularizer='L1L2')(s)
    c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu')(c1)
    do1 = tf.keras.layers.Dropout(0.15)(c1)

    m1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(do1)

    c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu', activity_regularizer='L1L2')(m1)
    c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu')(c2)
    do2 = tf.keras.layers.Dropout(0.15)(c2)

    m2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(do2)

    c3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu', activity_regularizer='L1L2')(m2)
    c3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu')(c3)
    c3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu')(c3)
    do3 = tf.keras.layers.Dropout(0.15)(c3)

    m3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(do3)

    c4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu', activity_regularizer='L1L2')(m3)
    c4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu')(c4)
    c4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu')(c4)
    do4 = tf.keras.layers.Dropout(0.15)(c4)

    m4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(do4)

    c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu', activity_regularizer='L1L2')(m4)
    c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu')(c5)
    c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu')(c5)
    do5 = tf.keras.layers.Dropout(0.15)(c5)

    f1 = tf.keras.layers.Flatten()(do5)

    d1 = tf.keras.layers.Dense(units=4096, activation='relu')(f1)
    d2 = tf.keras.layers.Dense(units=1024, activation='relu')(d1)

    d = tf.keras.layers.Dense(units=action_space, activation='linear')(d2)

    model = tf.keras.models.Model(inputs=[input], outputs=[d])
    
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"]) 

    # model.summary()
    return model

class DQNAgent:
    def __init__(self, chrome_path):
        self.env = DinoEnv(chrome_path=chrome_path)
        self.state_size = self.env.observation_space.shape
        #self.state_size = (120, 120, 4)
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 128
        self.train_start = 1000

        # create main model
        self.Target_model = OurModel(input_shape=self.state_size, action_space = self.action_size) 
        self.Train_model = OurModel(input_shape=self.state_size, action_space = self.action_size) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    # to do
    # implement the epsilon-greedy policy
    def act(self, state):
        p = np.random.uniform() 
        if p < self.epsilon: 
          action = self.env.action_space.sample() 
        else: 
          q = self.Train_model.predict(state[np.newaxis,:]) 
          action = np.argmax(q) 
        return action 

    # to do
    # implement the Q-learning
    def replay(self): 
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        #state = np.zeros((self.batch_size, self.state_size))
        state = np.zeros((self.batch_size, 120, 120, 4))
        #next_state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, 120, 120, 4))
        action, reward, done, targets = [], [], [], [] 

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):
            state[i] = minibatch[i][0] 
            next_state[i] = minibatch[i][3] 
            action.append(minibatch[i][1]) 
            reward.append(minibatch[i][2]) 
            done.append(minibatch[i][4]) 

        # compute value function of current state (call it target) and value function of next state (call it target_next)
        for i in range(self.batch_size):
            
            target = self.Train_model.predict(state[i][np.newaxis,:]) 
            target = target[0] 
            target_next = self.Target_model.predict(next_state[i][np.newaxis,:]) 
            target_next = target_next[0] 

            # correction on the Q value for the action used,
            # if done[i] is true, then the target should be just the final reward
            if not done[i]: 
                # else, use Bellman Equation
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # target = max_a' (r + gamma*Q_target_next(s', a'))

                q_next = np.max(target_next) 
                new_q = reward[i] + self.gamma*q_next 
            else:
                new_q = reward[i] 

            target[action[i]] = new_q 
            targets.append(target) 

        # Train the Neural Network with batches where target is the value function
        targets = np.asarray(targets) 
        self.Train_model.fit(state, targets, batch_size=self.batch_size, verbose=0)
        #self.memory = deque(maxlen=2000)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def training(self):
        max = 0 
        total_r = [] 
        count = 50
        start = time.time() 
        #self.env.Env_Start()

        for e in range(self.EPISODES):
            time.sleep(1.5)
            state = self.env.reset()
            done = False
            i = 0
            
            while not done:
                # if you have graphic support, you can render() to see the animation. 
                #self.env.render()

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                    
                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                i += 1
                if done:  
                    if i > max: 
                      max = i 
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")

                    end = time.time() 
                    elapse = np.abs(start-end) 
                    hour = elapse/3600 
                    minute = np.abs(elapse - (math.floor(hour) * 3600))/60 
                    seconds = np.abs(minute - math.floor(minute)) * 60 
                    #print(f"\repisode: {e+1}/{self.EPISODES}, score: {i}, max score: {max}, e: {round(self.epsilon, 4)}, time: {timestampStr}, elapsed time: {math.floor(hour)} hours, {math.floor(minute)} minutes, {math.floor(seconds)} seconds", end='', flush=True) 
                    print(f"episode: {e+1}/{self.EPISODES}, score: {i}, max score: {max}, e: {round(self.epsilon, 4)}, time: {timestampStr}, elapsed time: {math.floor(hour)} hours, {math.floor(minute)} minutes, {math.floor(seconds)} seconds") 
                    total_r.append(i) 

                    self.replay() 
                    if e > count: 
                        count += e 
                        self.Target_model.set_weights(self.Train_model.get_weights()) 

        epi = np.linspace(0, self.EPISODES, self.EPISODES) 
        plt.plot(epi, total_r) 
        plt.xlabel('Episodes') 
        plt.ylabel('Total Reward') 
        plt.show()

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	agent = DQNAgent(chrome_path=path)
	agent.training()