import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=512, num_layers=3, dropout=0.2, if_init=False):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.layers = nn.Sequential(*layers)
        self.softmax = self._softmax
        if if_init:
            self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _softmax(self, x):
        batch_size = x.shape[0]
        x = torch.softmax(x.view(batch_size, -1), dim=1)
        x = x[:, 1].view(-1, 1)
        return x

    def forward(self, x):
        x = self.layers(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = MLP(if_init=True)
    input = torch.randn(32, 300)

    print(input.shape)
    print(model(input))
