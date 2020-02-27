import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge
import meshplot as mp
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mp.offline()

dataset = ModelNet(root="C:/Users/vervl/Data/ModelNet/utils", name="10",
                   pre_transform=FaceToEdge(remove_faces=False), train=True)
test_dataset = ModelNet(root="C:/Users/vervl/Data/ModelNet/Test", name="10",
                        pre_transform=FaceToEdge(remove_faces=False), train=False)

print("cuda available: {}".format(torch.cuda.is_available()))

print("len dataset: {}".format(len(dataset)))
print("len test dataset: {}".format(len(test_dataset)))
print("data: {}".format(dataset[0]))
print("data y: {}".format(dataset[0].y.item()))
print("data y: {}".format(dataset[3990].y.item()))

test = dataset[0]
v = test.pos.numpy()
f = np.transpose(test.face.numpy())
print("faces: {}".format(test.face.size()))
print("vertices size: {}".format(v.shape))
print("faces size: {}".format(f.shape))

# mp.plot(v, f, v[:, 0], filename=RESULT_PATH + "test")
print(test.is_undirected())


def nb_edges(index, edges):
    filtered = np.array(
        [edge[0] == index or edge[1] == index for edge in edges]
    )
    return filtered.sum()


def nb_edges_tensor(index, edges):
    mask1 = edges[:, 0] == index
    return torch.sum(mask1)


def nb_edges_per_vertex(nb_vertices, edges):
    nbs = torch.empty(nb_vertices, dtype=torch.int64)
    for i in range(nb_vertices):
        nbs[i] = nb_edges_tensor(i, edges)
    return nbs


def edges_vertex_histogram(objects):
    data = []
    for i in range(len(objects)):
        obj = objects[i]
        data.append(nb_edges_per_vertex(obj.pos.size(0),
                                        torch.transpose(obj.edge_index, 0, 1)))
    cat = torch.cat(data)
    return cat


def edges_vertex_histogram_cuda(objects):
    data = []
    for i in range(len(objects)):
        obj = objects[i]
        data.append(nb_edges_per_vertex(obj.pos.size(0),
                                        torch.transpose(obj.edge_index, 0, 1)).to(DEVICE))
    cat = torch.cat(data)
    return cat


edges_tensor = torch.tensor([[1, 2], [1, 3], [4, 1]])
print(edges_tensor)
print(nb_edges_tensor(1, edges_tensor))

print(torch.arange(0, 5))
print(nb_edges_per_vertex(5, edges_tensor))

start = timer()
hist = edges_vertex_histogram(dataset[:5])
end = timer()
print("time: {}".format(end-start))
print(hist)

start = timer()
hist = edges_vertex_histogram_cuda(dataset[:10])
end = timer()
print("time cuda: {}".format(end-start))

plt.hist(hist.cpu().numpy(), bins=torch.max(hist).item())
plt.title("Nb edges / vertex")
plt.show()





