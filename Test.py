import torch
from dataset.functionality import ShapeNetFunctionality as snf
from torch_geometric.datasets import ShapeNet
import math


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NB_VERTICES = 2300

cloud1 = torch.tensor(
    [[1, 1, 1],
     [-1, -1, -1],
     [0, 0, 0]],
    dtype=torch.float
)
cloud2 = torch.tensor(
    [[1, 0, 0],
     [-1, 0, 0],
     [-1, 1, 0]],
    dtype=torch.float
)

dataset = ShapeNet(root="C:/Users/vervl/Data/ShapeNet/utils", train=True,
                   categories=["Airplane"])
filtered = snf.filter_data(dataset, NB_VERTICES)
resampled = snf.simple_resample_to_minimum(filtered)
test1 = resampled[5]
print(test1.pos.size())
test2 = resampled[10]
print(test2.pos.size())

x = [
    [1, 2, 3],
    [4, 5, 6]
]
y = [
    [7, 8, 9],
    [9, 8, 7]
]
print(x + y)

print(math.ceil(math.sqrt(453)))







