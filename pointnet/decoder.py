import torch.nn as nn
from Upsample import Upsample


class Decoder(nn.Module):
    def __init__(self, nb_upsample1, nb_upsample2):
        super(Decoder, self).__init__()
        self.up1 = Upsample(256, nb_upsample1, 64)
        self.up2 = Upsample(64, nb_upsample2, 16)
        self.up3 = Upsample(16, nb_upsample2, 3, final=True)

    def forward(self, feats):

        # feats = nb_batch x 1024
        up1 = self.up1(feats)
        # up1 = (nb_batch * nb_upsample1) x 256
        up2 = self.up2(up1)
        # up2 = (nb_batch * nb_upsample1 * nb_upsample2) x 64
        up3 = self.up3(up2)
        # up3 = (nb_batch * nb_upsample1 * nb_upsample2^2) x 3

        return up3



