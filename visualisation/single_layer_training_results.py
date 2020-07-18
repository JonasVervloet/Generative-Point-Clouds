import torch
from torch_geometric.data import Batch
import meshplot as mp
import numpy as np

from dataset.primitive_shapes import PrimitiveShapes
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder

# PATH VARIABLES + VARIA
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
LOAD_NAME = "ParameterReduction4/"
LOAD_PATH = RESULT_PATH + LOAD_NAME
SAVE_NAME = "TrainingResults4/"
SAVE_PATH = RESULT_PATH + SAVE_NAME
NB_EPOCHS = 100
NB_TESTS = 10

# DATASET VARIABLES
NB_POINTS = 3600
SPHERE = PrimitiveShapes.generate_spheres_dataset(1, NB_POINTS, False)[0]
CUBE = PrimitiveShapes.generate_cubes_dataset(1, NB_POINTS, False)[0]
CYLINDER = PrimitiveShapes.generate_cylinders_dataset(1, NB_POINTS, False)[0]
PYRAMID = PrimitiveShapes.generate_pyramids_dataset(1, NB_POINTS, False)[0]
TORUS = PrimitiveShapes.generate_tori_dataset(1, NB_POINTS, False)[0]
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
LAT_SIZE = 4

# ENCODER AND DECODER
ENCODER = NeighborhoodEncoder(
    nbs_features=[32, 64, 64],
    nbs_features_global=[64, 32, LAT_SIZE],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=LAT_SIZE,
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
    nb_neighbors=NBS_NEIGHS[-1]
)

print("SCRIPT STARTED")
print("DATASET PREP")
mp.offline()

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


def auto_encode_neighborhoods(data_obj, name):
    batch = Batch(pos=data_obj.pos, batch=torch.zeros(data_obj.pos.size(0), dtype=torch.int64))
    input_points, cluster_in, output_points, cluster_out = net(batch)
    data_obj_points_np = data_obj.pos.numpy()

    plot = None
    for i in range(NB_TESTS):
        current_input = input_points[cluster_in == i].detach().numpy()
        current_output = output_points[cluster_out == i].detach().numpy()

        points = np.concatenate((current_input, current_output))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (current_input.shape[0], 1)),
             np.tile([0, 0, 1.0], (current_output.shape[0], 1)))
        )
        if plot is None:
            plot = mp.subplot(
                data_obj_points_np, c=data_obj_points_np[:, 0],
                s=[2 * NB_TESTS, 1, 2 * i], shading={"point_size": 0.2}
            )
        else:
            mp.subplot(
                data_obj_points_np, c=data_obj_points_np[:, 0], s=[2 * NB_TESTS, 1, 2 * i],
                data=plot, shading={"point_size": 0.2}
            )
        plot.rows[2 * i][0].add_points(
            current_input, shading={"point_size": 0.2}
        )

        mp.subplot(points, c=colors, s=[2 * NB_TESTS, 1, 2 * i + 1],
                   data=plot, shading={"point_size": 0.2})

    plot.save(SAVE_PATH + "training_result_{}".format(name))


auto_encode_neighborhoods(SPHERE, "sphere")
auto_encode_neighborhoods(CUBE, "cube")
auto_encode_neighborhoods(CYLINDER, "cylinder")
auto_encode_neighborhoods(PYRAMID, "pyramid")
auto_encode_neighborhoods(TORUS, "torus")


def reconstruct_shape(data_obj, name):
    batch = Batch(pos=data_obj.pos, batch=torch.zeros(data_obj.pos.size(0), dtype=torch.int64))
    input_points, cluster_in, output_points, cluster_out = net(batch)
    input_points = input_points.detach().numpy()
    output_points = output_points.detach().numpy()
    data_obj_points_np = data_obj.pos.numpy()

    plot = mp.subplot(
        data_obj_points_np, c=data_obj_points_np[:, 0],
        s=[3, 1, 0], shading={"point_size": 0.2}
    )
    color_in = np.tile([1.0, 0, 0], (input_points.shape[0], 1))
    mp.subplot(
        input_points, c=color_in, data=plot,
        s=[3, 1, 1], shading={"point_size": 0.2}
    )
    color_out = np.tile([0, 0, 1.0], (output_points.shape[0], 1))
    mp.subplot(
        output_points, c=color_out, data=plot,
        s=[3, 1, 2], shading={"point_size": 0.2}
    )

    plot.save(SAVE_PATH + "reconstruction_{}".format(name))


reconstruct_shape(SPHERE, "sphere")
reconstruct_shape(CUBE, "cube")
reconstruct_shape(CYLINDER, "cylinder")
reconstruct_shape(PYRAMID, "pyramid")
reconstruct_shape(TORUS, "torus")