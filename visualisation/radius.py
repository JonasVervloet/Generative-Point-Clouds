import torch_geometric.nn as gnn
import torch
import numpy as np
import meshplot as mp
from dataset.primitives import PrimitiveShapes as ps
import matplotlib.pyplot as plt

# PATH VARIABLES
RESULT_PATH = "D:/Documenten/Results/Structured/"
NAME = "Visualization/"
PATH = RESULT_PATH + NAME

# DATASET VARIABLES
DATASET_SIZE = 50
NB_POINTS = 3600
NORMALS = False

# SINGLE LAYER NETWORK VARIABLES.
NB_LAYERS = 1
FINAL_LAYER = False
NBS_NEIGHBOURS = [25]
MAX_NEIGHBORS = 500
INTERVAL = np.arange(0.1, 0.31, 0.01)
NB_DECIMALS = 2


def analyse_dataset(dataset, name):
    """
    Analyses the given dataset and saves the results with the given name.
        An average number of neighbors plot will be made, where for each
        radius in INTERVAL, the average number of neighbors will be
        computed. Also, for each radius in INTERVAL, a histogram will be
        made that represents the distribution of number of neighbors for
        that radius.
    :param dataset: The dataset that will be analysed.
    :param name: The name which is used to name all the files that will
                    be saved.
    """
    nbs_neighbors = get_nbs_neighbors_dataset(dataset)
    assert(nbs_neighbors.shape[0] == len(INTERVAL))

    average_nbs = np.average(nbs_neighbors, axis=1)
    radius = None
    ideal_nb = NBS_NEIGHBOURS[-1]
    for i in range(len(average_nbs)):
        if average_nbs[i] >= ideal_nb:
            neigh_low = average_nbs[i - 1]
            neigh_high = average_nbs[i]
            alfa = (ideal_nb - neigh_low) / (neigh_high - neigh_low)

            radius_low = INTERVAL[i - 1]
            radius_step = INTERVAL[i] - radius_low
            radius = radius_low + alfa * radius_step

            break
    print("optimal radius: {}".format(radius))

    save_nbs = nbs_neighbors < 25
    save_sum = np.sum(save_nbs, axis=1)
    plot_avg_neigbors(
        average_nbs, name=name,
        ideal_radius=radius, save_index=np.argmax(save_sum==0)
    )

    for i in range(len(INTERVAL)):
        plot_histogram(
            nbs_neighbors[i],
            name,
            INTERVAL[i]
        )


def get_nbs_neighbors_dataset(dataset):
    """
    Get for each shape in the dataset and for each radius in INTERVAL
        the number of neighbors in each neighborhood.
    :param dataset: The dataset used to compute the neighborhoods.
    :return: A numpy array of dimension [len(INTERVAL),
                len(dataset) * number neighborhoods per shape]
    """
    print("computing nbs_neighbors")
    total_nbs = None
    for shape in dataset:
        nbs = get_nbs_neighbors(shape.pos)

        if total_nbs is None:
            total_nbs = nbs
        else:
            total_nbs = np.concatenate((total_nbs, nbs), axis=1)

    print(total_nbs.shape)
    return total_nbs


def get_nbs_neighbors(input_points):
    """
    For this point cloud, get the number of neighbors in each
        neighborhood. The neigborhoods are computed according
        to the global information.
        - NB_LAYERS
        - FINAL_LAYERS
        - NBS_NEIGHBORS
        This is done for each radius in INTERVAL.
    :param input_points: The input pointcloud that will be
                examined.
    :return: A numpy array of dimension [len(INTERVAL),
                number of neighborhoods per shape]
    """
    assert(not FINAL_LAYER)

    points, fps_points = get_neighborhood(input_points)

    total_nbs = None
    for radius in INTERVAL:
        rad_cluster, rad_inds = gnn.radius(
            points, fps_points, r=radius,
            max_num_neighbors=MAX_NEIGHBORS
        )

        nbs = []
        for i in range(torch.max(rad_cluster) + 1):
            nbs.append(
                len(rad_cluster[rad_cluster==i])
            )
        nbs_np = np.array([nbs])
        if total_nbs is None:
            total_nbs = nbs_np
        else:
            total_nbs = np.concatenate((total_nbs, nbs_np), axis=0)

    return total_nbs


def get_neighborhood(points):
    """
    Get the neighborhoods for the given points according to
        the global information.
        - NB_LAYERS
        - FINAL_LAYERS
        - NBS_NEIGHBORS
    :param points: The point cloud for which the neighborhoods
                will be computed
    :return: Set of points and a set of fps points. The fps points
                form the center of the neighborhoods. The set of
                points can be used to sample a neighborhood around
                these center points.
    """
    current_points = points
    current_fps_points = points

    for i in range(NB_LAYERS):
        current_points = current_points

        ratio = 1 / NBS_NEIGHBOURS[i]
        fps_inds = gnn.fps(current_points, ratio=ratio)
        current_fps_points = current_points[fps_inds]

    return current_points, current_fps_points


def plot_avg_neigbors(avg_neighbors, ideal_radius, save_index, name):
    """
    Plot the given average number of neighbors and save according
        the given name.
    :param avg_neighbors: The average number of neighbors for
                eacht radius in INTERVAL.
    :param name: A general name that will be extended to reflect
                the plot of average number of neighbors.
    """
    print("Average number of neigbors {}: layer {}".format(name, NB_LAYERS))
    assert(len(avg_neighbors) == len(INTERVAL))
    np.save(
        PATH + name + "_avg_nb_neighbors.npy", np.array(avg_neighbors)
    )

    plt.clf()
    plt.plot(INTERVAL, avg_neighbors)
    plt.legend([name])
    plt.plot(
        (INTERVAL[0], ideal_radius),
        (25, 25),
        'r'
    )
    plt.plot(
        (ideal_radius, ideal_radius),
        (avg_neighbors[0], 25),
        'r'
    )
    plt.plot(
        (INTERVAL[0], INTERVAL[save_index]),
        (avg_neighbors[save_index], avg_neighbors[save_index]),
        'g'
    )
    plt.plot(
        (INTERVAL[save_index], INTERVAL[save_index]),
        (avg_neighbors[0], avg_neighbors[save_index]),
        'g'
    )
    plt.title("Average number of neighbors {}: layer {}".format(name, NB_LAYERS))
    plt.savefig(PATH + "{}_layer{}_avg_nb_neighbors_plot".format(name, NB_LAYERS))


def plot_histogram(nbs_neighbors, name, radius):
    """
    Plot the given distribution of number of neighbors and save
        according to the given name and radius.
    :param nbs_neighbors: The number of neighbors of neighborhoods
                sampled with the given radius.
    :param name: A general name that will be extended to reflect
                the histogram of number of neighbors for this radius.
    :param radius: The radius that was used to sample the neighborhoods
                that led to nbs_neighbors
    """
    radius = round(radius, 2)
    print("Histogram {}: layer {}, radius {}".format(name, NB_LAYERS, radius))
    plt.clf()
    plt.hist(nbs_neighbors, bins=range(75))
    plt.title("Histogram {}: layer {}, radius {}".format(name, NB_LAYERS, radius))
    plt.savefig(PATH + "{}_layer{}_histogram_radius_{}.png".format(name, NB_LAYERS, radius))


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
print(INTERVAL)

print("CREATING DATASETS")
print("spheres")
spheres = ps.generate_spheres_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("cubes")
cubes = ps.generate_cubes_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("cylinders")
cylinders = ps.generate_cylinders_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("pyramids")
pyramids = ps.generate_pyramids_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("torus")
tori = ps.generate_tori_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("full")
full = spheres + cubes + cylinders + pyramids + tori

print("ANALYSING DATASETS")
print("spheres")
analyse_dataset(spheres, "sphere")
print("cubes")
analyse_dataset(cubes, "cube")
print("cylinders")
analyse_dataset(cylinders, "cylinder")
print("pyramids")
analyse_dataset(pyramids, "pyramid")
print("tori")
analyse_dataset(tori, "torus")
print("full")
analyse_dataset(full, "full")





