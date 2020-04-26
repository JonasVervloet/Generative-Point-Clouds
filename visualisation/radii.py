import torch_geometric.nn as gnn
import torch
import numpy as np
import meshplot as mp
from dataset.primitive_shapes import PrimitiveShapes as ps
import matplotlib.pyplot as plt

# PATH VARIABLES
RESULT_PATH = "D:/Documenten/Results/Visualisations/"
NAME = "Visalisation1/"
PATH = RESULT_PATH + NAME

# DATASET VARIABLES
DATASET_SIZE = 20
NB_POINTS = 3600
NORMALS = False

# SINGLE LAYER NETWORK VARIABLES.
NB_LAYERS = 1
FINAL_LAYER = False
NBS_NEIGHBOURS = [25]
INTERVAL = np.arange(0.1, 0.3, 0.005)


def radius_plot(dataset, name):
    avgs = []
    for radius in INTERVAL:
        print(radius)
        nbs = []
        for shape in dataset:
            nbs += get_nb_neighbors(shape.pos, radius)

        avgs.append(sum(nbs) / len(nbs))

    np.save(
        PATH + name + "_avg_nb_neighbors.npy", np.array(avgs)
    )
    plt.plot(INTERVAL, avgs)
    plt.legend([name])
    plt.title("Average number of neighbors: {}".format(name))
    plt.savefig(PATH + name + "_avg_nb_neighbors_plot")
    plt.show()


def get_nbs_neighbors_dataset(dataset):
    total_nbs = None
    for shape in dataset:
        nbs = get_nbs_neighbors(shape.pos)

        if total_nbs is None:
            total_nbs = nbs
        else:
            total_nbs = np.concatenate((total_nbs, nbs), axis=1)

    return total_nbs


def get_nbs_neighbors(points):
    assert(not FINAL_LAYER)

    current_points = points
    current_fps_points = points

    for i in range(NB_LAYERS):
        current_points = current_points

        ratio = 1/NBS_NEIGHBOURS[i]
        fps_inds = gnn.fps(current_points, ratio=ratio)
        current_fps_points = current_points[fps_inds]

    total_nbs = None
    for radius in INTERVAL:
        rad_cluster, rad_inds = gnn.radius(
            current_points, current_fps_points, r=radius
        )

        nbs = []
        for i in range(torch.max(rad_cluster) + 1):
            nbs.append(
                len(rad_cluster[rad_cluster==i])
            )
        nbs_np = np.array(nbs)
        if total_nbs is None:
            total_nbs = nbs_np
        else:
            total_nbs = np.concatenate((total_nbs, nbs_np), axis=0)

    return total_nbs


def create_radius_hist(dataset, nb_neighs, radius, title, path, second_layer=False, nb_neighs2=16):
    ratio = 1 / nb_neighs
    hist = []

    for shape in dataset:
        if not second_layer:
            hist += get_nb_neighbours(shape.pos, ratio, radius)
        else:
            ratio2 = 1/nb_neighs2
            hist += get_nb_neighbours2(shape.pos, ratio, ratio2, radius)

    print(len(hist))
    np_hist = np.array(hist)
    plt.hist(np_hist, bins=range(35))
    plt.title(title)
    plt.savefig(path)
    plt.show()


def plot_neighbourhoods(data, nb_neighs1, nb_neighs2, radius1, radius2, path):
    pos = data.pos
    np_pos = pos.numpy()

    fps_inds = gnn.fps(pos, ratio=1/nb_neighs1)
    fps_points = pos[fps_inds]

    rad_cluster, rad_inds = gnn.radius(pos, fps_points, r=radius1)
    rad_points = pos[rad_inds]

    fps_inds2 = gnn.fps(fps_points, ratio=1/nb_neighs2)
    fps_points2 = fps_points[fps_inds2]

    rad2_cluster, rad2_inds = gnn.radius(fps_points, fps_points2, r=radius2)
    rad2_points = fps_points[rad2_inds]

    plot = mp.subplot(
        np_pos, c=np_pos[:, 0], s=[3, 1, 0],
        shading={"point_size": 0.4}
    )
    nb_fps_points = fps_points.size(0)
    nb_fps_points_half = nb_fps_points / 2
    for i in range(nb_fps_points):
        np_cluster = rad_points[rad_cluster == i].numpy()
        if i < nb_fps_points_half:
            g = 0.5 + i / nb_fps_points_half
        else:
            g = 0.5 - (i - nb_fps_points_half) / nb_fps_points_half
        color = np.tile([1.0 - (i / nb_fps_points), g, i / nb_fps_points], (np_cluster.shape[0], 1))
        if i == 0:
            mp.subplot(
                np_cluster, c=color,
                data=plot, s=[3, 1, 1], shading={"point_size": 0.4}
            )
        else:
            plot.rows[1][0].add_points(
                np_cluster, c=color, shading={"point_size": 0.4}
            )

    nb_fps_points = fps_points2.size(0)
    nb_fps_points_half = nb_fps_points / 2
    for i in range(nb_fps_points):
        np_cluster = rad2_points[rad2_cluster == i].numpy()
        if i < nb_fps_points_half:
            g = 0.5 + i / nb_fps_points_half
        else:
            g = 0.5 - (i - nb_fps_points_half) / nb_fps_points_half
        color = np.tile([1.0 - i / nb_fps_points, g, i / nb_fps_points], (np_cluster.shape[0], 1))
        if i == 0:
            mp.subplot(
                np_cluster, c=color,
                data=plot, s=[3, 1, 2], shading={"point_size": 0.4}
            )
        else:
            plot.rows[2][0].add_points(
                np_cluster, c=color, shading={"point_size": 0.4}
            )

    plot.save(path)


print("CHECKING GLOBAL VARIABLES")
assert(NB_LAYERS == len(NBS_NEIGHBOURS))

print("CREATING DATASETS")
print("spheres")
spheres = ps.generate_spheres(DATASET_SIZE, NB_POINTS, NORMALS)
print("cubes")
cubes = ps.generate_cubes(DATASET_SIZE, NB_POINTS, NORMALS)
print("cylinders")
cylinders = ps.generate_cylinders(DATASET_SIZE, NB_POINTS, NORMALS)
print("pyramids")
pyramids = ps.generate_pyramids(DATASET_SIZE, NB_POINTS, NORMALS)
print("torus")
torus = ps.generate_tori(DATASET_SIZE, NB_POINTS, NORMALS)
print("full")
full = spheres + cubes + cylinders + pyramids + torus

# print("CREATING HISTOGRAMS")
# if not SECOND_LAYER:
#     radius = RADIUS
# else:
#     radius = RADIUS2
# print("radius: {}".format(radius))
# print("spheres")
# create_radius_hist(spheres, NB_NEIGHBOURS, radius,
#                    "Spheres: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_spheres", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("cubes")
# create_radius_hist(cubes, NB_NEIGHBOURS, radius,
#                    "Cubes: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_cubes", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("cylinders")
# create_radius_hist(cylinders, NB_NEIGHBOURS, radius,
#                    "Cylinders: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_cylinders", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("pyramids")
# create_radius_hist(pyramids, NB_NEIGHBOURS, radius,
#                    "Pyramids: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_pyramids", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("torus")
# create_radius_hist(torus, NB_NEIGHBOURS, radius,
#                    "Torus: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_torus", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("full")
# create_radius_hist(full, NB_NEIGHBOURS, radius,
#                    "Full: {} neighbours, {} radius".format(NB_NEIGHBOURS, radius),
#                    RESULT_PATH + "radius_hist_full", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)

# print("CREATING RADIUS PLOTS")
# if not SECOND_LAYER:
#     interval = np.arange(0.1, 0.3, 0.005)
# else:
#     interval = np.arange(0.5, 1.5, 0.05)
# print("interval: {}".format(interval))
# print("spheres")
# radius_plot(spheres, interval, NB_NEIGHBOURS, "Spheres: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_spheres", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("cubes")
# radius_plot(cubes, interval, NB_NEIGHBOURS, "Cubes: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_cubes", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("cylinders")
# radius_plot(cylinders, interval, NB_NEIGHBOURS, "Cylinders: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_cylinders", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("pyramids")
# radius_plot(pyramids, interval, NB_NEIGHBOURS, "Pyramids: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_pyramids", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("torus")
# radius_plot(torus, interval, NB_NEIGHBOURS, "Torus: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_torus", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)
# print("full")
# radius_plot(full, interval, NB_NEIGHBOURS, "Full: {} neighbours".format(NB_NEIGHBOURS2),
#             RESULT_PATH + "radius_plot_full", second_layer=SECOND_LAYER, nb_neighs2=NB_NEIGHBOURS2)

print("CREATING NEIGHBOURHOODS")
mp.offline()
print("spheres")
plot_neighbourhoods(spheres[0], NB_NEIGHBOURS, NB_NEIGHBOURS2,
                    RADIUS, RADIUS2, RESULT_PATH + "neighbourhoods_spheres")
print("cubes")
plot_neighbourhoods(cubes[0], NB_NEIGHBOURS, NB_NEIGHBOURS2,
                    RADIUS, RADIUS2, RESULT_PATH + "neighbourhoods_cubes")
print("cylinders")
plot_neighbourhoods(cylinders[0], NB_NEIGHBOURS, NB_NEIGHBOURS2,
                    RADIUS, RADIUS2, RESULT_PATH + "neighbourhoods_cylinders")
print("pyramids")
plot_neighbourhoods(pyramids[0], NB_NEIGHBOURS, NB_NEIGHBOURS2,
                    RADIUS, RADIUS2, RESULT_PATH + "neighbourhoods_pyramids")
print("torus")
plot_neighbourhoods(torus[0], NB_NEIGHBOURS, NB_NEIGHBOURS2,
                    RADIUS, RADIUS2, RESULT_PATH + "neighbourhoods_torus")



