#!/usr/bin/env python3

import random
import numpy as np
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env \
    import KukaDiverseObjectEnv

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()  # turn interactive mode on

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)


def get_screen():
    global stacked_screens
    # returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env._get_observation().transpose((2, 0, 1))
    # convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)


def evaluate():
    scores_window = collections.deque(maxlen=100)  # last 100 scores

    # load the model
    checkpoint = torch.load(save_model_path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # evaluate the model
    for i_episode in range(episode):
        env.reset()
        state = get_screen()
        stacked_states = collections.deque(
            STACK_SIZE*[state], maxlen=STACK_SIZE)
        for t in count():
            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
            # select and perform an action
            action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
            _, reward, done, _ = env.step(action.item())
            # observe new state
            next_state = get_screen()
            stacked_states.append(next_state)
            if done:
                break
        print("Episode: {0:d}, reward: {1}".format(
            i_episode+1, reward), end="\n")


if __name__ == '__main__':
    STACK_SIZE = 5
    preprocess = T.Compose(
        [T.ToPILImage(),
         T.Grayscale(num_output_channels=1),
         T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
         T.ToTensor()])

    isRendersEval = True
    env = KukaDiverseObjectEnv(
        renders=isRendersEval,
        isDiscrete=True,
        removeHeightHack=False,
        maxSteps=20,
        isTest=True)
    env.cid = p.connect(p.DIRECT)
    env.reset()

    # get screen size so that we can initialize layers correctly based on shape
    # returned from pybullet (48, 48, 3).
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    # get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    save_model_path = osp.join(
        osp.dirname(__file__),
        '..',
        'data',
        'result',
        'robot',
        'policy_dqn.pt')
    episode = 3
    evaluate()
