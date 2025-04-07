import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, if_init=False):
        super().__init__()
        self.W_n = nn.Linear(2 * in_dim, out_dim)
        self.W_m = nn.Linear(in_dim, 1)
        self.W_f = nn.Linear(in_dim, 1)
        if if_init:
            self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, h, adj):
        h_neighbors = torch.bmm(adj, h) / (adj.sum(dim=2, keepdim=True) + 1e-6)
        m = torch.sigmoid(self.W_m(h) + self.W_f(h_neighbors))
        h_cat = torch.cat([h, h_neighbors], dim=-1)
        h_new = F.relu(self.W_n(h_cat) * m)
        return h_new, m

class MaskedGNN(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=512, out_dim=300, num_layers=3, if_init=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MaskedGNNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(MaskedGNNLayer(hidden_dim, hidden_dim))
        self.layers.append(MaskedGNNLayer(hidden_dim, out_dim))
        if if_init:
            self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            layer._init_weights()

    def forward(self, h, adj):
        mask_weights = []
        for layer in self.layers:
            h, m = layer(h, adj)
            mask_weights.append(m)
        return h

def readout(h: torch.Tensor) -> torch.Tensor:
    return torch.mean(h, dim=1, keepdim=True).squeeze(1)

if __name__ == "__main__":
    N = 10
    in_dim = 300
    hidden_dim = 512
    out_dim = 300
    num_layers = 3
    torch.manual_seed(199)

    node_features = torch.rand(N, in_dim).repeat(32, 1, 1)
    adj_matrix = torch.randint(0, 2, (N, N), dtype=torch.float32).repeat(32, 1, 1)

    gnn = MaskedGNN(in_dim, hidden_dim, out_dim, num_layers, if_init=True)
    output_features = gnn(node_features, adj_matrix)

    print("output_features:", output_features.shape)
    print("output_features:", readout(output_features).shape)
