import torch_geometric.nn as gnn
import torch
import numpy as np
import meshplot as mp
from dataset.primitive_shapes import PrimitiveShapes as ps
import matplotlib.pyplot as plt

RESULT_PATH = "D:/Documenten/Results/Visualisations/"


def radius_plot(dataset, interval, nb_neighs, title, path, second_layer=False, nb_neighs2=16):
    ratio = 1 / nb_neighs
    avgs = []

    for radius in interval:
        print(radius)
        nbs = []
        for shape in dataset:
            if not second_layer:
                nbs += get_nb_neighbours(shape.pos, ratio, radius)
            else:
                ratio2 = 1/nb_neighs2
                nbs += get_nb_neighbours2(shape.pos, ratio, ratio2, radius)

        avgs.append(sum(nbs) / len(nbs))

    plt.plot(interval, avgs)
    plt.legend(['radius'])
    plt.title(title)
    plt.savefig(path)
    plt.show()


def get_nb_neighbours(points, ratio, radius):
    fps_inds = gnn.fps(points, ratio=ratio)
    fps_points = points[fps_inds]

    rad_cluster, rad_inds = gnn.radius(points, fps_points, radius)
    nb_clusters = torch.max(rad_cluster)

    nbs = []
    for i in range(nb_clusters + 1):
        nbs.append(
            len(rad_cluster[rad_cluster==i])
        )

    return nbs


def get_nb_neighbours2(points, ratio, ratio2, radius):
    fps_inds = gnn.fps(points, ratio=ratio)
    fps_points = points[fps_inds]

    fps_inds2 = gnn.fps(fps_points, ratio=ratio2)
    fps_points2 = fps_points[fps_inds2]

    rad_cluster, rad_inds = gnn.radius(fps_points, fps_points2, radius)
    nb_clusters = torch.max(rad_cluster)

    nbs = []
    for i in range(nb_clusters + 1):
        nbs.append(
            len(rad_cluster[rad_cluster == i])
        )

    return nbs


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


NB_NEIGHBOURS = 25
NB_NEIGHBOURS2 = 16
RADIUS = 0.23
RADIUS2 = 1.1
SECOND_LAYER = True

print("CREATING DATASETS")
print("spheres")
spheres = ps.generate_dataset(20, 3600, [True, False, False, False, False])
print("cubes")
cubes = ps.generate_dataset(20, 3600, [False, True, False, False, False])
print("cylinders")
cylinders = ps.generate_dataset(20, 3600, [False, False, True, False, False])
print("pyramids")
pyramids = ps.generate_dataset(20, 3600, [False, False, False, True, False])
print("torus")
torus = ps.generate_dataset(20, 3600, [False, False, False, False, True])
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



