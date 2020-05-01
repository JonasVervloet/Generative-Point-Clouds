import torch
from torch_geometric.data import DataLoader
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder


# PATH VARIABLES + VARIA
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
LOAD_NAME = "ParameterReduction3/"
LOAD_PATH = RESULT_PATH + LOAD_NAME
SAVE_NAME = "Interpolation3/"
SAVE_PATH = RESULT_PATH + SAVE_NAME
NB_EPOCHS = 100
NB_TESTS = 5
NB_INTERPOLS = 10
EPSILON = 1e-10

# DATASET VARIABLES
NB_POINTS = 3600
NORMALS = False
SIZE = 1
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
    nbs_features_global=[64, 32, 8],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=8,
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
    nb_neighbors=NBS_NEIGHS[-1]
)

print("SCRIPT STARTED")
print("DATASET PREP")
mp.offline()
# SPHERES
spheres = PrimitiveShapes.generate_spheres_dataset(
    nb_objects=SIZE,
    nb_points=NB_POINTS,
    normals=NORMALS
)
sphere_loader = DataLoader(
    dataset=spheres,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# CUBES
cubes = PrimitiveShapes.generate_cubes_dataset(
    nb_objects=SIZE,
    nb_points=NB_POINTS,
    normals=NORMALS
)
cube_loader = DataLoader(
    dataset=cubes,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# CYLINDERS
cylinders = PrimitiveShapes.generate_cylinders_dataset(
    nb_objects=SIZE,
    nb_points=NB_POINTS,
    normals=NORMALS
)
cylinder_loader = DataLoader(
    dataset=cylinders,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# PYRAMIDS
pyramids = PrimitiveShapes.generate_pyramids_dataset(
    nb_objects=SIZE,
    nb_points=NB_POINTS,
    normals=NORMALS
)
pyramid_loader = DataLoader(
    dataset=pyramids,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# TORI
tori = PrimitiveShapes.generate_tori_dataset(
    nb_objects=SIZE,
    nb_points=NB_POINTS,
    normals=NORMALS
)
tori_loader = DataLoader(
    dataset=tori,
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


def encode(network, loader):
    latent_list = []
    for batch in loader:
        latent_list.append(
            network.encode(batch)
        )

    return torch.cat(latent_list, dim=0)


print("ENCODING")
spheres_encoded = encode(net, sphere_loader)
cubes_encoded= encode(net, cube_loader)
cylinders_encoded = encode(net,cylinder_loader)
pyramids_encoded = encode(net, pyramid_loader)
tori_encoded = encode(net, tori_loader)
shapes_list = [
    [spheres_encoded, "sphere"],
    [cubes_encoded, "cube"],
    [cylinders_encoded, "cylinder"],
    [pyramids_encoded, "pyramid"],
    [tori_encoded, "torus"]
]


def interpolate(network, encoded1, name1, encoded2, name2):
    rand_perm1 = encoded1[torch.randperm(encoded1.size(0))[:NB_TESTS]]
    print(rand_perm1.size())
    rand_perm2 = encoded2[torch.randperm(encoded2.size(0))[:NB_TESTS]]
    print(rand_perm2.size())

    nb = 0
    for latent_vec1 in rand_perm1:
        for latent_vec2 in rand_perm2:
            step = (latent_vec2 - latent_vec1) / (NB_INTERPOLS - 1)
            plot = None

            for i in range(NB_INTERPOLS):
                current_latent_vec = latent_vec1 + i * step
                decoded, cluster = network.decode(current_latent_vec)
                dec_np = decoded.detach().numpy()

                if plot is None:
                    plot = mp.subplot(
                        dec_np, c=dec_np[:, 0],
                        s=[NB_INTERPOLS, 1, i], shading={"point_size": 0.3}
                    )
                else:
                    mp.subplot(
                        dec_np, c=dec_np[:, 0], data=plot,
                        s=[NB_INTERPOLS, 1, i], shading={"point_size": 0.3}
                    )

            plot.save(SAVE_PATH + "interpolation_"+name1+"_"+name2+"{}".format(nb))
            nb += 1


print("INTERPOLATION")
for i in range(len(shapes_list) - 1):
    print(shapes_list[i][1])
    for j in range(i+1, len(shapes_list)):
        interpolate(
            net, shapes_list[i][0], shapes_list[i][1],
            shapes_list[j][0], shapes_list[j][1]
        )
