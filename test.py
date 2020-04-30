import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

RESULT_PATH = "D:/Documenten/Results/"

arange = torch.arange(10)

mean = torch.rand(8)

repeated = mean.repeat((10, 1))

print(arange.size())
print(arange)
print(mean.size())
print(mean)
print(repeated.size())
print(repeated)

repeated[:, 2] = arange
print(repeated.size())
print(repeated)













