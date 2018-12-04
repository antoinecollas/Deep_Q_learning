import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, agent_history_length, nb_actions):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=agent_history_length, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.l1 = nn.Linear(in_features=3136, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=nb_actions)

    def forward(self, x):
       x = self.conv(x)
       x = x.view(x.shape[0], -1)
       x = self.l1(x)
       x = self.l2(x)
       return x

class MLP(nn.Module):
    def __init__(self, obs_space, nb_actions):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=obs_space, out_features=128),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=nb_actions),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x