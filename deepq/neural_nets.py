import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, agent_history_length, nb_actions):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=agent_history_length, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.l1 = nn.Linear(in_features=3136, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=nb_actions)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class CNN2(nn.Module):
    def __init__(self, agent_history_length, nb_actions):
        super(CNN2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=agent_history_length, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),            
        )
        self.l1 = nn.Linear(in_features=512, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=nb_actions)

        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                print(m)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
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