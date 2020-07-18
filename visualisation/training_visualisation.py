import torch
from torch_geometric.data import Batch
import meshplot as mp
import numpy as np


def auto_encode_neighborhoods(network, data_obj, nb_tests, path, name):
    mp.offline()

    batch = Batch(pos=data_obj.pos, batch=torch.zeros(data_obj.pos.size(0), dtype=torch.int64))
    results = network(batch)
    input_points = results[0][0]
    input_clusters = results[1][0]
    output_points = results[2][0]
    output_clusters = results[3][0]

    data_obj_points_np = data_obj.pos.numpy()

    plot = None
    for i in range(nb_tests):
        current_input = input_points[input_clusters == i].detach().numpy()
        current_output = output_points[output_clusters == i].detach().numpy()

        points = np.concatenate((current_input, current_output))
        colors = np.concatenate(
            (np.tile([1.0, 0, 0], (current_input.shape[0], 1)),
             np.tile([0, 0, 1.0], (current_output.shape[0], 1)))
        )
        if plot is None:
            plot = mp.subplot(
                data_obj_points_np, c=data_obj_points_np[:, 0],
                s=[2 * nb_tests, 1, 2 * i], shading={"point_size": 0.2}
            )
        else:
            mp.subplot(
                data_obj_points_np, c=data_obj_points_np[:, 0], s=[2 * nb_tests, 1, 2 * i],
                data=plot, shading={"point_size": 0.2}
            )
        plot.rows[2 * i][0].add_points(
            current_input, shading={"point_size": 0.2}
        )

        mp.subplot(points, c=colors, s=[2 * nb_tests, 1, 2 * i + 1],
                   data=plot, shading={"point_size": 0.2})

    plot.save(path + "training_result_{}".format(name))


def reconstruct_shape(network, data_obj, path, name):
    mp.offline()

    batch = Batch(pos=data_obj.pos, batch=torch.zeros(data_obj.pos.size(0), dtype=torch.int64))
    results = network(batch)
    input_points = results[0][0]
    output_points = results[2][0]

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

    plot.save(path + "reconstruction_{}".format(name))