import torch
import torch.nn as nn
import torch.nn.functional as F
from relative_layer.decoder import SimpleRelativeDecoder


class MiddleLayerDecoder(nn.Module):
    def __init__(self):
        super(MiddleLayerDecoder, self).__init__()

        self.decoder = SimpleRelativeDecoder(5, 10, 20, 20)
        self.fc1 = nn.Linear(28, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, encoded, feats):

        # encoded = nb_batch x 5
        # feats = nb_batch x 25

        decoded = self.decoder(encoded)
        # decoded = (nb_batch * 20) x 3

        repeat = feats.repeat_interleave(20, dim=0)
        # repeat = (nb_batch * 20) x 25

        concat = torch.cat([decoded, repeat], 1)
        # concat = (nb_batch * 20) x 28

        fc1 = F.relu(self.fc1(concat))
        # fc1 = (nb_batch * 20) x 10

        fc2 = F.relu(self.fc2(fc1))
        # fc2 = (nb_batch * 20) x 5

        return decoded, fc2


