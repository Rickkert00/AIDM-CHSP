from torch import nn

class BoundPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3785, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
