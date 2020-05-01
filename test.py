import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

RESULT_PATH = "D:/Documenten/Results/"

arr1 = np.array(
    [[1, 2, 3, 4, 5, 6]]
)
arr2 = np.array(
    [4, 5, 6, 7, 8, 9]
)
arr2 = [arr2]

concat = np.concatenate((arr1, arr2), axis=0)
print(concat)













