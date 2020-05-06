import torch
from torch_geometric.data import DataLoader
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder


# PATH VARIABLES + VARIA
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
LOAD_NAME = "ParameterReduction2/"
LOAD_PATH = RESULT_PATH + LOAD_NAME
SAVE_NAME = "Interpolation2/"
SAVE_PATH = RESULT_PATH + SAVE_NAME
NB_EPOCHS = 100
NB_TESTS = 10
EPSILON = 1e-10

# DATASET VARIABLES
NB_POINTS = 3600
SPHERE = True
CUBE = True
CYLINDER = True
PYRAMID = True
TORUS = True
SHAPES = [SPHERE, CUBE, CYLINDER, PYRAMID, TORUS]
NORMALS = False
SIZE = 5
BATCH_SIZE = 5

# SINGLE LAYER NETWORK VARIABLES.
CUDA = False
DEVICE = torch.device(
    "cuda:0" if CUDA and torch.cuda.is_available() else "cpu"
)
NB_LAYERS = 1
NBS_NEIGHS = [25]
FINAL_LAYER = False
RADIUS = 0.23

# ENCODER AND DECODER
ENCODER = NeighborhoodEncoder(
    nbs_features=[32, 64, 64],
    nbs_features_global=[64, 32, 16],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=16,
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
    nb_neighbors=NBS_NEIGHS[-1]
)

print("SCRIPT STARTED")
print("DATASET PREP")
mp.offline()
dataset = PrimitiveShapes.generate_dataset(
    nb_objects=SIZE, nb_points=NB_POINTS,
    shapes=SHAPES,normals=NORMALS
)
loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("NETWORK SETUP")
net = SingleLayerNetwork(
    nb_layers=NB_LAYERS,
    nbs_neighbors=NBS_NEIGHS,
    final_layer=FINAL_LAYER,
    radius=RADIUS,
    device=DEVICE
)
net.set_encoder(ENCODER)
net.set_decoder(DECODER)
net.load_state_dict(
    torch.load(LOAD_PATH + "model_epoch{}.pt".format(NB_EPOCHS))
)
net.to(DEVICE)
net.eval()

print("ENCODING")
latent_list = []
for batch in loader:
    encoded = net.encode(batch)
    latent_list.append(encoded)

latent = torch.cat(latent_list, dim=0)
print(latent.size())

print("min")
latent_min, inds = torch.min(latent, dim=0)
print(latent_min)

print("max")
latent_max, inds = torch.max(latent, dim=0)
print(latent_max)

print("mean")
latent_mean = torch.mean(latent, dim=0)
print(latent_mean)

# print("INTERPOLATION")
# for index in range(latent_mean.size(0)):
#     print("Index: {}".format(index))
#     maximum = latent_max[index].item()
#     minimum = latent_min[index].item()
#     print(maximum - minimum)
#     if (maximum - minimum) < EPSILON:
#         print("No interpolation possible: continue")
#         continue
#     interval = torch.arange(
#         minimum,
#         maximum,
#         (maximum-minimum)/(NB_TESTS)
#     )
# 
#     repeated = latent_mean.repeat((NB_TESTS, 1))
#     repeated[:, index] = interval
#     print(repeated.size())
# 
#     decoded, cluster = net.decode(repeated)
#     print(decoded.size())
#     print(cluster.size())
#     print(torch.max(cluster))
# 
#     plot = None
#     for i in range(torch.max(cluster) + 1):
#         decoded_points = decoded[cluster==i]
#         dec_np = decoded_points.detach().numpy()
# 
#         if plot is None:
#             plot = mp.subplot(
#                 dec_np, c=dec_np[:, 0],
#                 s=[NB_TESTS, 1, i], shading={"point_size": 0.3}
#             )
#         else:
#             mp.subplot(
#                 dec_np, c=dec_np[:, 0], data=plot,
#                 s=[NB_TESTS, 1, i], shading={"point_size": 0.3}
#             )
# 
#     plot.save(SAVE_PATH + "feature_visualization{}".format(index))


