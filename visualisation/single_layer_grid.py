import torch
import torch_geometric.nn as gnn
import numpy as np
import meshplot as mp

from relative_layer.grid_deform_decoder import GridDeformationDecoder
from relative_layer.simple_layer import SimpleRelativeLayer
from dataset.primitive_shapes import PrimitiveShapes as ps


def visualize_grid_deformation(shape, nb_neighs, radius, net, path, nb_tests):
    net.eval()
    ae = net.ae
    encoder = net.ae.enc
    decoder = net.ae.dec

    pos = shape.pos
    np_pos = pos.numpy()

    fps_inds = gnn.fps(pos, ratio=1 / nb_neighs)
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

        cluster = torch.zeros((neighs_rel.size(0)), dtype=torch.int64)
        encoded = encoder(neighs_rel, cluster)
        grid, deform1, deform2 = decoder.eval(encoded)
        grid_np = grid.detach().numpy()
        grid_np = np.stack([grid_np[:, 0], np.zeros(grid_np.shape[0]), grid_np[:, 1]], axis=-1)
        deform1_np = deform1.detach().numpy()
        deform2_np = deform2.detach().numpy()

        output = ae(neighs_rel, cluster)
        output_np = output.detach().numpy()

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
            neighs_np, shading={"point_size": 0.4}
        )

        points = np.concatenate((neighs_rel_np, deform2_np))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (neighs_rel_np.shape[0], 1)),
             np.tile([0, 0, 1.0], (deform2_np.shape[0], 1)))
        )
        mp.subplot(
            points, c=colors, s=[nb_tests * 2, 1, 2 * i + 1],
            data=plot, shading={"point_size": 0.3}
        )
        plot.rows[2*i + 1][0].add_points(
            grid_np, shading={"point_size": 0.2, "point_color": "green"}
        )
        plot.rows[2 * i + 1][0].add_lines(
            grid_np, deform2_np, shading={"line_color": "black"}
        )

    plot.save(path)


NB_TESTS = 20
NB_EPOCHS = 60
NB_NEIGHBOURS = 25
RADIUS = 0.195
NB_POINTS = 3600
RESULT_PATH = "D:/Documenten/Results/LearningRate2/GirdDeformationMoreExpressive/"

mp.offline()

print("STARTING EVALUATING GRID DEFORMATION")

print("DATASET PREP")
sphere = ps.generate_dataset(1, NB_POINTS, [True, False, False, False, False])[0]
cube = ps.generate_dataset(1, NB_POINTS, [False, True, False, False, False])[0]
cylinder = ps.generate_dataset(1, NB_POINTS, [False, False, True, False, False])[0]
pyramid = ps.generate_dataset(1, NB_POINTS, [False, False, False, True, False])[0]
torus = ps.generate_dataset(1, NB_POINTS, [False, False, False, False, True])[0]

print("Visualize different shapes")
net = SimpleRelativeLayer(NB_NEIGHBOURS, 80, 40, 20, RADIUS)
net.set_decoder(GridDeformationDecoder(NB_NEIGHBOURS, 20, 40, 80))
net.load_state_dict(
    torch.load(RESULT_PATH + "model_epoch{}.pt".format(NB_EPOCHS))
)
visualize_fn = visualize_grid_deformation
print("sphere")
visualize_fn(sphere, NB_NEIGHBOURS, RADIUS, net, RESULT_PATH+"sphere_visualisation", NB_TESTS)
print("cube")
visualize_fn(cube, NB_NEIGHBOURS, RADIUS, net, RESULT_PATH+"cube_visualisation", NB_TESTS)
print("cylinder")
visualize_fn(cylinder, NB_NEIGHBOURS, RADIUS, net, RESULT_PATH+"cylinder_visualisation", NB_TESTS)
print("pyramid")
visualize_fn(pyramid, NB_NEIGHBOURS, RADIUS, net, RESULT_PATH+"pyramid_visualisation", NB_TESTS)
print("torus")
visualize_fn(torus, NB_NEIGHBOURS, RADIUS, net, RESULT_PATH+"torus_visualisation", NB_TESTS)