import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from relative_layer.decoder import SimpleRelativeDecoder


class ComposedLayerDecoder(nn.Module):
    def __init__(self):
        super(ComposedLayerDecoder, self).__init__()

        self.decoder = SimpleRelativeDecoder(5, 10, 20, 20)
        self.fc1 = nn.Linear(28, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, encoded, feats):
        print("decoder!")

        # encoded = nb_batch x 5
        # feats = nb_batch x 25

        decoded = self.decoder(encoded)
        # decoded = (nb_batch * 20) x 3
        print(decoded.size())

        repeat = feats.repeat_interleave(20, dim=0)
        # repeat = (nb_batch * 20) x 25
        print(repeat.size())

        concat = torch.cat([decoded, repeat], 1)
        # concat = (nb_batch * 20) x 28
        print(concat.size())

        fc1 = F.relu(self.fc1(concat))
        # fc1 = (nb_batch * 20) x 10
        print(fc1.size())

        fc2 = F.relu(self.fc2(fc1))
        # fc2 = (nb_batc * 20) x 5
        print(fc2.size())

        return decoded, fc2


