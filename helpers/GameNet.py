import torch
import random
import numpy as np
from copy import deepcopy


class GameNet(torch.nn.Module):
    def __init__(self):
        super(GameNet, self).__init__()
        self.fc1 = torch.nn.Linear(11, 14)
        self.fc2 = torch.nn.Linear(14, 4)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x.detach()

    def mutate(self, p=.15):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            for index, weight in enumerate(state_dict[key]):
                if random.uniform(0, 1) <= p:
                    delta = random.uniform(-.5, .5)
                    state_dict[key][index] += delta
        return self.load_state_dict(state_dict)

    def crossover(self, other, p=.3):
        state_dict0 = deepcopy(self.state_dict())
        state_dict1 = deepcopy(other.state_dict())
        keys = list(state_dict1.keys())
        for key in keys:
            if random.uniform(0, 1) <= p:
                x = np.random.randint(1, len(state_dict0[key]))
                state_dict0[key][:x], state_dict1[key][:x] = state_dict1[key][:x].clone(), state_dict0[key][:x].clone()

        child_net0 = GameNet()
        child_net1 = GameNet()
        child_net0.load_state_dict(state_dict0)
        child_net1.load_state_dict(state_dict1)
        return child_net0.cuda(), child_net1.cuda()

    def save_model(self, path="model"):
        torch.save(self, path)

    @staticmethod
    def load_model(path="model"):
        model = torch.load(path)
        model.eval()
        return model
