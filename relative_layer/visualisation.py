import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
import meshplot as mp

import numpy as np

from dataset.primitives import PrimitiveShapes
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.rotation_invariant_layer import RotationInvariantLayer

# PATH VARIABLES
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
NAME = "Network3/"
PATH = RESULT_PATH + NAME

# EPOCH + DATASET + TESTS
NB_EPOCHS = 50
NB_POINTS = 3600
NB_TESTS = 20

# SINGLE LAYER NETWORK VARIABLES
NB_LAYERS = 1
NBS_NEIGHS = [25]
FINAL_LAYER = False
RADIUS = 0.23

# ENCODER AND DECODER
ENCODER = NeighborhoodEncoder(
    nbs_features=[80, 40, 20],
    pool_index=1,
    mean=False
)
DECODER = NeighborhoodDecoder(
    nbs_features=[20, 40, 80],
    unpool_index=2,
    nb_neighbors=NBS_NEIGHS[-1]
)


def visualize_learing_process(pos, neighs, rels, cluster, model, nb_vis, path):
    pos_np = pos.numpy()
    neighs_np = neighs.numpy()
    neigh_rel_np = rels.numpy()

    mp.offline()
    plot = mp.subplot(
        pos_np, c=pos_np[:, 0],
        s=[nb_vis + 3, 1, 0], shading={"point_size": 0.3}
    )
    mp.subplot(
        neighs_np, s=[nb_vis + 3, 1, 1],
        data=plot, shading={"point_size": 0.3}
    )
    mp.subplot(
        pos_np, c=pos_np[:, 0], s=[nb_vis + 3, 1, 2],
        data=plot, shading={"point_size": 0.3}
    )
    plot.rows[2][0].add_points(
        neighs_np, c=np.tile([1.0, 0.5, 0.5], (neighs_np.shape[0], 1)),
        shading={"point_size": 0.4}
    )

    for i in range(nb_vis):
        epoch = i * 5
        print("Epoch: {}".format(epoch))

        model.load_state_dict(
            torch.load(path + "model_epoch{}.pt".format(epoch))
        )
        model.eval()
        ae = model.ae

        relative_out = ae(rels, cluster)
        output_np = relative_out.detach().numpy()

        points = np.concatenate((neigh_rel_np, output_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (neigh_rel_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (output_np.shape[0], 1)))
        )

        mp.subplot(
            points, c=colors, s=[nb_vis + 3, 1, i + 3],
            data=plot, shading={"point_size": 0.3}
        )

    plot.save(path + "learning_process")


def visualize_neighbourhoods(shape, model, nb_tests, path):
    mp.offline()
    model.eval()

    pos = shape.pos
    batch_obj = Batch(pos=pos, batch=torch.zeros(pos.size(0), dtype=torch.int64))
    np_pos = pos.numpy()

    input_points, cluster_in, output_points, cluster_out = model(batch_obj)

    plot_created = False
    for i in range(nb_tests):
        inputs = input_points[cluster_in == i]
        outputs = output_points[cluster_out == i]
        inputs_np = inputs.numpy()
        outputs_np = outputs.detach().numpy()

        if not plot_created:
            plot = mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests * 2, 1, 2 * i],
                shading={"point_size": 0.3}
            )
            plot_created = True
        else:
            mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests * 2, 1, 2 * i],
                data=plot, shading={"point_size": 0.3}
            )
        plot.rows[2 * i][0].add_points(
            inputs_np, shading={"point_size": 0.4}
        )

        points = np.concatenate((inputs_np, outputs_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (inputs_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (outputs_np.shape[0], 1)))
        )
        mp.subplot(
            points, c=colors, s=[nb_tests * 2, 1, 2 * i + 1],
            data=plot, shading={"point_size": 0.3}
        )

    plot.save(path)


def visualize_neighbourhoods2(shape, nb_neighs, radius, net, path, nb_tests, nb_neighs2=16):
    mp.offline()

    net.eval()
    ae = net.ae

    pos = shape.pos
    np_pos = pos.numpy()

    fps_inds = gnn.fps(pos, ratio=1/nb_neighs)
    fps_points = pos[fps_inds]
    fps_points_np = fps_points.numpy()

    fps_inds2 = gnn.fps(fps_points, ratio=1/nb_neighs2)
    fps_points2 = fps_points[fps_inds2]

    rad_cluster, rad_inds = gnn.radius(fps_points, fps_points2, r=radius)
    rad_points = fps_points[rad_inds]
    rad_midpoints = fps_points2[rad_cluster]
    relatives = (rad_points - rad_midpoints) / radius

    plot_created = False

    nb = min(nb_tests, max(rad_cluster) + 1)

    for i in range(nb):
        neighs = rad_points[rad_cluster == i]
        neighs_np = neighs.numpy()

        neighs_rel = relatives[rad_cluster == i]
        neighs_rel_np = neighs_rel.numpy()

        cluster = torch.zeros((neighs_rel.size(0)), dtype=torch.int64)
        output = ae(neighs_rel, cluster)
        output_np = output.detach().numpy()

        if not plot_created:
            plot = mp.subplot(
                fps_points_np, c=fps_points_np[:, 0], s=[nb * 2, 1, 2 * i],
                shading={"point_size": 0.3}
            )
            plot_created = True
        else:
            mp.subplot(
                fps_points_np, c=fps_points_np[:, 0], s=[nb * 2, 1, 2 * i],
                data=plot, shading={"point_size": 0.3}
            )
        plot.rows[2 * i][0].add_points(
            neighs_np, shading={"point_size": 0.4}
        )

        points = np.concatenate((neighs_rel_np, output_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (neighs_rel_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (output_np.shape[0], 1)))
        )
        mp.subplot(
            points, c=colors, s=[nb * 2, 1, 2 * i + 1],
            data=plot, shading={"point_size": 0.3}
        )

    plot.save(path)


def visualize_decoder2(shape, nb_neighs, radius, net, path, nb_tests):
    mp.offline()

    net.eval()
    enc = net.ae.enc
    dec = net.ae.dec

    pos = shape.pos
    np_pos = pos.numpy()

    fps_inds = gnn.fps(pos, ratio= 1/nb_neighs)
    fps_points = pos[fps_inds]

    rad_cluster, rad_inds = gnn.radius(pos, fps_points, r=radius)
    rad_points = pos[rad_inds]
    rad_midpoints = fps_points[rad_cluster]
    relatives = (rad_points - rad_midpoints) / radius

    plot_created = False

    for i in range(nb_tests):
        neighs = rad_points[rad_cluster == i]
        neighs_np = neighs.numpy()

        neighs_rel = relatives[rad_cluster == i]
        neighs_rel_np = neighs_rel.numpy()

        cluster = rad_cluster[rad_cluster == i]
        encoded = enc(neighs_rel, cluster)
        rot_inv, rotated = dec.eval(encoded)
        rot_inv_np = rot_inv.detach().numpy()
        rotated_np = rotated.detach().numpy()

        if not plot_created:
            plot = mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests*2, 1, 2 * i],
                shading={"point_size": 0.3}
            )
            plot_created = True
        else:
            mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests * 2, 1, 2 * i],
                data=plot, shading={"point_size": 0.3}
            )
        plot.rows[2 * i][0].add_points(
            neighs_np, shading={"point_size": 0.4}
        )

        points = np.concatenate((neighs_rel_np, rot_inv_np, rotated_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (neighs_rel_np.shape[0], 1)),
             np.tile([0, 1.0, 0], (rot_inv_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (rotated_np.shape[0], 1)))
        )
        mp.subplot(
            points, c=colors, s=[nb_tests * 2, 1, 2 * i + 1],
            data=plot, shading={"point_size": 0.3}
        )

    plot.save(path)


def visualize_rot_inv_layer(shape, nb_neighs, radius, net, path, nb_tests):
    mp.offline()

    net.eval()
    enc = net.enc
    dec = net.dec

    pos = shape.pos
    norm = shape.norm
    np_pos = pos.numpy()

    fps_inds = gnn.fps(pos, ratio=1 / nb_neighs)
    fps_points = pos[fps_inds]
    fps_normals = norm[fps_inds]

    rad_cluster, rad_inds = gnn.radius(pos, fps_points, r=radius)
    rad_points = pos[rad_inds]
    rad_normals = norm[rad_inds]
    rad_midpoints = fps_points[rad_cluster]
    midpoints_normals = fps_normals[rad_cluster]
    relatives = (rad_points - rad_midpoints) / radius

    rot_inv_feats = RotationInvariantLayer.inv_features(relatives, rad_normals, midpoints_normals)

    plot_created = False

    for i in range(nb_tests):
        neighs = rad_points[rad_cluster == i]
        neighs_np = neighs.numpy()

        neighs_rel = relatives[rad_cluster == i]
        neighs_rel_np = neighs_rel.numpy()

        inv_feat = rot_inv_feats[rad_cluster == i]

        cluster = torch.zeros((neighs_rel.size(0)), dtype=torch.int64)
        encoded, angles = enc(neighs_rel, inv_feat, cluster)
        rot_inv, rotated = dec.eval(torch.cat([angles, encoded], dim=-1))
        rot_inv_np = rot_inv.detach().numpy()
        rotated_np = rotated.detach().numpy()

        if not plot_created:
            plot = mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests*2, 1, 2 * i],
                shading={"point_size": 0.3}
            )
            plot_created = True
        else:
            mp.subplot(
                np_pos, c=np_pos[:, 0], s=[nb_tests * 2, 1, 2 * i],
                data=plot, shading={"point_size": 0.3}
            )
        plot.rows[2 * i][0].add_points(
            neighs_np, shading={"point_size": 0.4}
        )

        points = np.concatenate((neighs_rel_np, rot_inv_np, rotated_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (neighs_rel_np.shape[0], 1)),
             np.tile([0, 1.0, 0], (rot_inv_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (rotated_np.shape[0], 1)))
        )
        mp.subplot(
            points, c=colors, s=[nb_tests * 2, 1, 2 * i + 1],
            data=plot, shading={"point_size": 0.3}
        )

    plot.save(path)


net = SingleLayerNetwork(
    nb_layers=NB_LAYERS,
    nbs_neighbors=NBS_NEIGHS,
    final_layer=FINAL_LAYER,
    radius=RADIUS
)
net.set_encoder(ENCODER)
net.set_decoder(DECODER)

print("STARTING EVALUATING")
print("DATASET PREP")
sphere = PrimitiveShapes.generate_dataset(1, NB_POINTS, [True, False, False, False, False], True)[0]
cube = PrimitiveShapes.generate_dataset(1, NB_POINTS, [False, True, False, False, False], True)[0]
cylinder = PrimitiveShapes.generate_dataset(1, NB_POINTS, [False, False, True, False, False], True)[0]
pyramid = PrimitiveShapes.generate_dataset(1, NB_POINTS, [False, False, False, True, False], True)[0]
torus = PrimitiveShapes.generate_dataset(1, NB_POINTS, [False, False, False, False, True], True)[0]

# shape = pyramid
#
# print("Visualize learning process")
#
# nb_evals = int(NB_EPOCHS / 5) + 1
#
# net = get_net()
#
# pos = shape.pos
#
# sample_inds = gnn.fps(pos, ratio=1/NB_NEIGHS)
# samples = pos[sample_inds]
#
# rad_cluster, rad_inds = gnn.radius(pos, samples, r=RADIUS)
# rad_points = pos[rad_inds]
# rad_midpoints = samples[rad_cluster]
# relatives = (rad_points - rad_midpoints) / RADIUS
#
# neighbourhood = rad_points[rad_cluster == 0]
# neighs_rel = relatives[rad_cluster == 0]
# cluster = rad_cluster[rad_cluster == 0]
#
# visualize_learing_process(pos, neighbourhood, neighs_rel, cluster, net, nb_evals, RESULT_PATH + NAME)

print("Visualize different shapes")
net.load_state_dict(
    torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(NB_EPOCHS))
)
visualize_fn = visualize_neighbourhoods
print("sphere")
visualize_fn(sphere, net, NB_TESTS, RESULT_PATH+NAME+"sphere_visualisation")
print("cube")
visualize_fn(cube, net, NB_TESTS, RESULT_PATH+NAME+"cube_visualisation")
print("cylinder")
visualize_fn(cylinder, net, NB_TESTS, RESULT_PATH+NAME+"cylinder_visualisation")
print("pyramid")
visualize_fn(pyramid, net, NB_TESTS, RESULT_PATH+NAME+"pyramid_visualisation")
print("torus")
visualize_fn(torus, net, NB_TESTS, RESULT_PATH+NAME+"torus_visualisation")