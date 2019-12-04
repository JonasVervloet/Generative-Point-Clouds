import torch
from PointCloudAE import PointCloudAE
from torch_geometric.datasets import ShapeNet
from DatasetFunctionality import ShapeNetFunctionality as snf
import meshplot as mp
import random
import numpy as np

RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"
NB_TESTS = 5

test = torch.tensor([[1, 2], [3, 4],
                     [5, 6], [7, 8],
                     [9, 10], [11, 12],
                     [13, 14], [14, 16]])
print(test)
print(test.reshape(-1))
print(test.reshape(4, -1))

# model = PointCloudAE()
# model.load_state_dict(torch.load(RESULT_PATH + "pointcloudvae_model.pt"))
# print(model)

# dataset = ShapeNet(root="./data/ShapeNet", categories=["Airplane"])
# print("length dataset: " + str(len(dataset)))
# print("Total number of nodes: {}".format(snf.get_total_nb_nodes(dataset)))
#
# filtered = snf.filter_data(dataset, 2300)
# resampled = snf.simple_resample_to_minimum(filtered)
# resampled = resampled[:100]
#
# test_samples = random.sample(range(len(resampled)), NB_TESTS)

# mp.offline()
# plot = 0
# for i in range(NB_TESTS):
#     print("TEST {}".format(i))
#     test = dataset[test_samples[i]].pos
#     result = model(test)
#     if not plot == 0:
#         mp.subplot(test, c=test[:, 0], data=plot)
#     else:
#         plot = mp.subplot(test, c=test[:, 0])
#     mp.subplot(result, c=result[:, 0], data=plot)
# plot.save(RESULT_PATH + "pointcloudvae_eval")

# test_samples = random.sample(range(len(dataset)), NB_TESTS)
#
# mp.offline()
# plot = 0
# for i in range(len(test_samples)):
#     print(i)
#     data = dataset[test_samples[i]]
#     v = data.pos.numpy()
#     if not i == 0:
#         mp.subplot(v, c=v[:, 0], s=[NB_TESTS, 1, i],
#                    data=plot, shading={"point_size": 0.1})
#     else:
#         plot = mp.subplot(v, c=v[:, 0], s=[NB_TESTS, 1, 0],
#                           shading={"point_size": 0.1})
#
#     print(data)
#     print(data.pos.size())
#     print(data.pos.numpy().shape)
#     print("")
#
# print(plot)
#
# plot.save(RESULT_PATH + "shapenet_airplane")





