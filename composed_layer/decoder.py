import torch
import torch.nn as nn
import torch.nn.functional as F
from relative_layer.decoder import SimpleRelativeDecoder


class MiddleLayerDecoder(nn.Module):
    def __init__(self, nb_feats_in1=20, nb_feats_in2=30,
                 nb_feats_out=20, nb_outputs=25):
        super(MiddleLayerDecoder, self).__init__()

        nb_feats_middle = int((nb_feats_in2 + nb_feats_out) / 2)

        self.decoder = SimpleRelativeDecoder(nb_feats_in1, 40, 80, nb_outputs)
        self.fc1 = nn.Linear(nb_feats_in2 + 3, nb_feats_middle)
        self.fc2 = nn.Linear(nb_feats_middle, nb_feats_out)

        self.nb_outputs = nb_outputs

    def forward(self, encoded, feats):

        # encoded = nb_cluster x nb_feats_in1
        # feats = nb_cluster x nb_feats_in2

        decoded = self.decoder(encoded)
        # decoded = (nb_cluster * nb_neighs) x 3

        repeat = feats.repeat_interleave(self.nb_outputs, dim=0)
        # repeat = (nb_cluster * nb_neighs) x nb_feats_in2

        concat = torch.cat([decoded, repeat], 1)
        # concat = (nb_batch * nb_neighs) x (nb_feats_in2 + 3)

        fc1 = F.relu(self.fc1(concat))
        # fc1 = (nb_batch * nb_neighs) x 10

        fc2 = F.relu(self.fc2(fc1))
        # fc2 = (nb_batch * nb_neighs) x nb_feats_out

        return decoded, fc2


