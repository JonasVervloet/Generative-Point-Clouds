from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from full_network.network_generator import NetworkGenerator
from dataset.primitive_shapes import PrimitiveShapes
from network_manager import NetworkManager
from loss_function import ChamferDistLossFullNetwork
import wandb

wandb.init(project="generative-point-clouds")

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

"""DATASET GENERATOR"""
train_size = 200
val_size = 20
nb_points = 3600
batch_size = 5
shuffle = True

dataset_generator = PrimitiveShapes(
    train_size, val_size,
    nb_points, batch_size,
    shuffle
)

"""NETWORK MANAGER"""
path = "D:/Documenten/Results/Structured/Test/TestWandB/"

optimizer = Adam
learning_rate = 0.001
weight_decay = 5e-4
loss_function = ChamferDistLossFullNetwork()

load = False
end_epoch = 50

manager = NetworkManager()
manager.set_path(path)
if not load:
    manager.set_optimizer(optimizer)
    manager.set_learning_rate(learning_rate)
    manager.set_weight_decay(weight_decay)
    manager.set_loss_function(loss_function)
    network = generator.generate_network()
    wandb.watch(network, log='all')
    manager.set_network(network)
    manager.set_dataset_generator(dataset_generator)
else:
    manager.load_from_file()
    network = manager.network
    wandb.watch(network, log='all')

manager.train(end_epoch, wandb_logging=True)

