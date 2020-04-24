import point_cloud_utils as pcu
import numpy as np
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes as ps


def generate_normals_for_dataset(dataset, path):
    points = []
    normals = []
    for data in dataset:
        points.append(data.pos.numpy())
        normals.append(data.norm.numpy())
    plot_normals(np.array(points), np.array(normals), path)


def plot_normals(points, normals, path):
    assert(points.shape == normals.shape)

    mp.offline()
    mp_created = False

    nb = len(points)
    for i in range(nb):
        pos = points[i]
        norm = normals[i]

        if not mp_created:
            plot = mp.subplot(
                pos, c=(norm + 1) * 0.5,
                s=[nb, 1, i], shading={"point_size": 0.3}
            )
            mp_created = True
        else:
            mp.subplot(
                pos, c=(norm + 1) * 0.5, data=plot,
                s=[nb, 1, i], shading={"point_size": 0.3}
            )

    plot.save(path)


RESULT_PATH = "D:/Documenten/Results/Visualisations/"

dataset = ps.generate_dataset(1, 3600, normals=True)

generate_normals_for_dataset(dataset, RESULT_PATH+"normals")