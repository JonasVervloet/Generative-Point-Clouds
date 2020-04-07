import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRelativeDecoder(nn.Module):
    def __init__(self, nb_feats_in, nb_feats1, nb_feats2, nb_outputs):
        super(SimpleRelativeDecoder, self).__init__()

        self.fc1 = nn.Linear(nb_feats_in, nb_feats1)
        self.fc2 = nn.Linear(nb_feats1, nb_feats2)
        self.fc3 = nn.Linear(nb_feats2, nb_feats2 * nb_outputs)
        # self.fc3 = nn.Linear(nb_feats2, nb_feats1 * nb_outputs)

        self.conv = nn.Conv1d(nb_feats2, 3, 1)
        # self.conv = nn.Conv1d(nb_feats1, 3, 1)

        self.feats2 = nb_feats2
        # self.feats1 = nb_feats1

    def forward(self, feats):
        # feats = nb_batch x nb_feats_in

        fc1 = F.relu(self.fc1(feats))
        # fc1 = nb_batch x nb_feats1

        fc2 = F.relu(self.fc2(fc1))
        # fc2 = nb_batch x nb_feats2

        fc3 = F.relu(self.fc3(fc2))
        # fc3 = nb_batch x (nb_feats2 * nb_outputs)

        resized = fc3.view(-1, self.feats2)
        # resized = fc3.view(-1, self.feats1)
        # resized = (nb_batch * nb_outputs) x nb_feats2

        squeezed = resized.unsqueeze(0)
        # squeezed = 1 x (nb_batch * nb_outputs) x nb_feats2

        transpose = squeezed.transpose(2, 1)
        # transpose = 1 x nb_feats2 x (nb_batch * nb_outputs)

        conv = torch.tanh(self.conv(transpose))
        # conv = 1 x 3 x (nb_batch * nb_outputs)

        out = conv.transpose(1, 2)
        # out = 1 x (nb_batch * nb_outputs) x 3

        return out[0]

