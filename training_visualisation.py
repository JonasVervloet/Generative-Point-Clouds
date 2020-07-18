import torch
from torch_geometric.data import Batch
from dataset.primitive_shapes import PrimitiveShapes
from network_manager import NetworkManager
import visualisation.training_visualisation as tvis
import numpy as np
import meshplot as mp

path = path = "D:/Documenten/Results/Structured/Test/SingleLayer/"
nb_tests = 9

manager = NetworkManager()
manager.set_path(path)
manager.load_from_file()

network = manager.network
print(network.to_string())

dataset_generator = manager.dataset_generator
nb_points = dataset_generator.nb_points

sphere = PrimitiveShapes.generate_spheres_dataset(1, nb_points, False)[0]
cube = PrimitiveShapes.generate_cubes_dataset(1, nb_points, False)[0]
cylinder = PrimitiveShapes.generate_cylinders_dataset(1, nb_points, False)[0]
pyramid = PrimitiveShapes.generate_pyramids_dataset(1, nb_points, False)[0]
torus = PrimitiveShapes.generate_tori_dataset(1, nb_points, False)[0]


tvis.auto_encode_neighborhoods(network, sphere, nb_tests, path, "sphere")
tvis.auto_encode_neighborhoods(network, cube, nb_tests, path, "cube")
tvis.auto_encode_neighborhoods(network, cylinder, nb_tests, path, "cylinder")
tvis.auto_encode_neighborhoods(network, pyramid, nb_tests, path, "pyramid")
tvis.auto_encode_neighborhoods(network, torus, nb_tests, path, "torus")

tvis.reconstruct_shape(network, sphere, path, "sphere")
tvis.reconstruct_shape(network, cube, path, "cube")
tvis.reconstruct_shape(network, cylinder, path, "cylinder")
tvis.reconstruct_shape(network, pyramid, path, "pyramid")
tvis.reconstruct_shape(network, torus, path, "torus")
