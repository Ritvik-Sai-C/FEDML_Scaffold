from torch import nn

class ANN(nn.Module):
    def __init__(self, input_dim, name, B, K, lr):
        super(ANN, self).__init__()
        self.name = name
        self.B = B
        self.K = K
        self.len = 0
        self.lr = lr
        self.loss = 0
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x
