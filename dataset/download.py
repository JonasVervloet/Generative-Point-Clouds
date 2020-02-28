from torch_geometric.datasets import ShapeNet
from dataset.functionality import ShapeNetFunctionality as snf

name = "Mug"

root = "C:/Users/vervl/Data/ShapeNet/utils/Mug"
categories = ["Mug"]

nb_bins = 30

dataset = ShapeNet(root=root, train=True, categories=categories)
print(dataset)
snf.histogram(dataset, name, nb_bins)