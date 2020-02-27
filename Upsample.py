import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, nb_input_feats, nb_outputs, nb_output_feats, final=False):
        super(Upsample, self).__init__()
        self.in_feats = nb_input_feats
        self.nb_outs = nb_outputs
        self.out_feats = nb_output_feats
        self.fc = nn.Linear(nb_input_feats, nb_outputs * nb_input_feats)
        self.conv1d = nn.Conv1d(nb_input_feats, nb_output_feats, 1)
        self.final = final

    def forward(self, feats):

        # feats = (nb_batch * nb_points) x nb_feats
        up = F.relu(self.fc(feats))
        # up = (nb_samples * nb_points) x (nb_feats * nb_outputs)
        resized = up.view(-1, self.in_feats)
        # resized = (nb_samples * nb_points * nb_outputs) x nb_feats
        squeezed = resized.unsqueeze(0)
        # squeeze = 1 x (nb_samples * nb_points * nb_outputs) x nb_feats
        transpose = squeezed.transpose(2, 1)
        # transpose = 1 x nb_feats x (nb_samples * nb_points * nb_outputs)
        conv1 = self.conv1d(transpose)
        if self.final:
            conv1 = torch.tanh(conv1)
        else:
            conv1 = F.relu(conv1)
        # conv1 = 1 x nb_out_feats x (nb_samples * nb_points * nb_outputs)
        n_feats = conv1.transpose(1, 2)
        # n_feats = 1 x (nb_samples * nb_points * nb_outputs) x nb_out_feats

        return n_feats[0]

