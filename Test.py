import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

from dataset.primitives import PrimitiveShapes as ps
import meshplot as mp

from relative_layer.decoder3 import RelativeDecoder2
from relative_layer.decoder import SimpleRelativeDecoder
from relative_layer.rotation_invariant_layer import RotationInvariantLayer
from relative_layer.encoder3 import RotationInvariantEncoder

RESULT_PATH = "D:/Documenten/Results/"

test = torch.tensor([1, 2, 3, 4, 5])
test2 = test.clone()
test2[0] = 5
print(test)
print(test2)












