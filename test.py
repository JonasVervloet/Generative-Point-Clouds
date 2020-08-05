import torch
from torch.utils.tensorboard import SummaryWriter
from full_network.network_generator import NetworkGenerator
from dataset.primitive_shapes import PrimitiveShapes
from loss_function import ChamferDistLossFullNetwork


"""NETWORK GENERATOR"""
complete_encoding = False
nb_layers = 1

nbs_neighbors = [25, 16, 9]
radii = [0.23, 1.3, 2.0]
latent_sizes = [8, 64, 128]

generator = NetworkGenerator()
generator.set_max_pooling()
generator.set_grid_decoder()
generator.disable_split_networks()
generator.deactivate_leaky_relu()

generator.set_nb_layers(nb_layers)
generator.set_nbs_neighbors(nbs_neighbors[:nb_layers])
generator.set_radii(radii[:nb_layers])
generator.set_latent_sizes(latent_sizes[:nb_layers])
if complete_encoding:
    generator.make_complete_encoding()
else:
    generator.make_incomplete_encoding()

network = generator.generate_network()

"""DATASET GENERATOR"""
train_size = 1
val_size = 1
nb_points = 3600
batch_size = 1
shuffle = True

dataset_generator = PrimitiveShapes(
    train_size, val_size,
    nb_points, batch_size,
    shuffle
)
dataset_generator.disable_cubes()
dataset_generator.disable_cylinders()
dataset_generator.disable_pyramids()
dataset_generator.disable_tori()
train_loader, val_loader = dataset_generator.generate_loaders()

loss_fn = ChamferDistLossFullNetwork()

writer = SummaryWriter('D:/DocumentenResults/Test/runs/test_graph')
for batch in train_loader:
    writer.add_graph(network, batch)
    break
    # result_list = network(batch)
    # loss = loss_fn(result_list)
    #
    # print()
    # print(loss)





