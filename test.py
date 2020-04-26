import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

RESULT_PATH = "D:/Documenten/Results/"

test1 = [
    [1, 2, 3, 4, 5],
]
arr1 = np.array(test1)
test2 = [
    [16, 17, 18, 19, 20]
]
arr2 = np.array(test2)
test3 = [
    [16, 17, 18, 19, 20]
]
arr3 = np.array(test3)

res1 = np.concatenate((arr1, arr2), axis=0)
res2 = np.concatenate((res1, arr3), axis=0)

print(res1)
print(res2)















