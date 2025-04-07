import torch.nn as nn

from .BiLSTM import BiLSTM
from .MaskedGNN import MaskedGNN, readout
from .MLP import MLP

class CombinedModel(nn.Module):
    def __init__(self, in_dim=50, if_init=False):
        super().__init__()
        self.bilstm = BiLSTM(in_dim=in_dim, if_init=if_init)
        self.gnn = MaskedGNN(in_dim=in_dim, out_dim=in_dim, if_init=if_init)
        self.mlp = MLP(in_dim=in_dim, if_init=if_init)

    def forward(self, x, adj):
        h = self.bilstm(x)
        gnn_out = self.gnn(h, adj)
        z = readout(gnn_out)
        result = self.mlp(z)
        return result

class CombinedLoss(nn.Module):
    def __init__(self, l2_weight=1e-3):
        super().__init__()
        self.l2_weight = l2_weight
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, model):
        loss = self.criterion(outputs, targets)
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        l2_loss = self.l2_weight * l2_norm
        total_loss = loss + l2_loss
        return total_loss
