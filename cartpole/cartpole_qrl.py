#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import gym
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import animation


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)
    anim.save(save_movie_path)
    plt.clf()
    plt.close()


class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states     # states are 4
        self.num_actions = num_actions   # actions are 2 (move to left/right)
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, observation, action, reward, observation_next):
        self.brain.update_Qtable(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        # Q function (table): number of divisions^(4 variables)
        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(NUM_DIZITIZED**self.num_states, self.num_actions))

    def bins(self, clip_min, clip_max, num):
        # converting observed states (continuous values) to discrete values
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        # converting observations (continuous values) to discrete values
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

    def update_Qtable(self, observation, action, reward, observation_next):
        # update Q table with Q learning
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = \
            self.q_table[state, action] + \
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        # gradually adopt only optimal action with the ε-greedy method
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # actions of 0, 1
        return action


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)  # generate agent

    def run(self):
        complete_episodes = 0  # add if CartPole stands after 195 steps
        episode_final = False  # final flag to finish trials

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()  # initialize environment
            episode_reward = 0  # reward in episode

            # loop for one episode
            for step in range(MAX_STEPS):

                if episode_final is True:
                    # add the image of this step to frames
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(observation, episode)

                # by doing action a_t, get s_{t+1} and r_{t+1}
                observation_next, reward_notuse, done, info_notuse = \
                    self.env.step(action)

                # calculate reward for the trials
                if done:  # if step was over 200 or pole tilted with
                    if step < 195:
                        reward = -1  # if CartPole fell over
                        self.complete_episodes = 0
                    else:
                        reward = 1   # if CartPole is still standing
                        self.complete_episodes = self.complete_episodes + 1
                else:
                    reward = 0

                episode_reward += reward  # add reward

                # using the state observation_next at step+1, update Q function
                self.agent.update_q_function(
                    observation, action, reward, observation_next)

                # update observation
                observation = observation_next

                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(
                        episode, step+1))
                    break

            if episode_final is True:
                # save and display movie
                display_frames_as_gif(frames)
                break

            if self.complete_episodes >= 10:
                print('Succeeded 10 times in a row!')
                frames = []
                episode_final = True  # next is final trial to display


if __name__ == '__main__':

    ENV = 'CartPole-v0'  # environment name used
    NUM_DIZITIZED = 6    # number of divisions
    GAMMA = 0.99         # time discount rate
    ETA = 0.5            # learning coefficient
    MAX_STEPS = 200      # number of steps in one trial
    NUM_EPISODES = 1000  # number of muximum trials
    save_movie_path = osp.join(
        osp.dirname(__file__),
        '..',
        'data',
        'result',
        'movie_cartpole.mp4')

    cartpole_env = Environment()
    cartpole_env.run()
