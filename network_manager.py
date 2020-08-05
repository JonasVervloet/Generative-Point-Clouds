import torch
import numpy as np
import datetime
from torch.optim import Adam
import wandb

from network_text_converter import NetworkTextFileConverter


class NetworkManager:
    def __init__(self):
        self.network_converter = None
        self.dataset_generator = None

        self.optimizer = Adam
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.loss_function = None

        self.network = None

        self.train_losses = []
        self.val_losses = []

        self.training_starts = []
        self.training_stops = []
        self.epoch_stops = [0]

    def set_path(self, path):
        self.network_converter = NetworkTextFileConverter(path)

    def set_network(self, network):
        self.network = network

    def set_dataset_generator(self, dataset_generator):
        self.dataset_generator = dataset_generator

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_training_starts(self, starts):
        self.training_starts = starts

    def set_training_stops(self, stops):
        self.training_stops = stops

    def set_epoch_stops(self, stops):
        self.epoch_stops = stops

    def train(self, end_epoch, wandb_logging=False):
        start_epoch = self.epoch_stops[-1]
        assert (end_epoch > start_epoch)
        assert(end_epoch % 5 == 0)

        if start_epoch == 0:
            self.network_converter.save_info_file(self)
        else:
            self.load_epoch(start_epoch)
        self.epoch_stops.append(start_epoch)

        print("generating data loaders...")
        train_loader, val_loader = self.dataset_generator.generate_loaders()
        print("data loaders generated")
        train_size, val_size = self.dataset_generator.get_sizes()
        optimizer = self.optimizer(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        print("TRAINING STARTING")
        self.training_starts.append(str(datetime.datetime.now()))
        self.training_stops.append(str(datetime.datetime.now()))
        for i in range(end_epoch + 1 - start_epoch):
            epoch = start_epoch + i
            print("epoch {}".format(epoch))

            if start_epoch != 0 and epoch == start_epoch:
                continue

            # training
            self.network.train()
            temp_loss = []
            for batch in train_loader:
                optimizer.zero_grad()
                output_list = self.network(batch)
                loss = self.loss_function(output_list)
                loss.backward()
                optimizer.step()
                temp_loss.append(loss.item())

            train_loss = sum(temp_loss) / train_size
            self.train_losses = np.append(self.train_losses, train_loss)
            print("train loss: {}".format(train_loss))

            # validation
            self.network.eval()
            temp_loss = []
            for val_batch in val_loader:
                output_list = self.network(batch)
                loss = self.loss_function(output_list)
                temp_loss.append(loss.item())

            val_loss = sum(temp_loss) / val_size
            self.val_losses = np.append(self.val_losses, val_loss)
            print("val loss: {}".format(val_loss))

            if wandb_logging:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    }
                )

            # saving
            if epoch % 5 == 0:
                self.epoch_stops[-1] = epoch
                self.training_stops[-1] = str(datetime.datetime.now())
                self.save(epoch)

    def save(self, epoch):
        self.network_converter.save_network(self.network, epoch)
        self.network_converter.save_losses(self.train_losses, epoch, train=True)
        self.network_converter.save_losses(self.val_losses, epoch, train=False)
        self.network_converter.save_loss_plot(self.train_losses, self.val_losses, epoch)
        self.network_converter.save_info_file(self)

    def load_epoch(self, epoch):
        self.network_converter.load_network(
            self.network, epoch
        )
        self.train_losses = self.network_converter.load_losses(
            epoch, train=True
        )
        self.val_losses = self.network_converter.load_losses(
            epoch, train=False
        )

    def to_string(self):
        assert(len(self.epoch_stops) == len(self.training_starts) + 1)
        assert(len(self.epoch_stops) == len(self.training_stops) + 1)

        string = "DATASET GENERATOR" + "\n"
        string += self.dataset_generator.to_string()
        string += "\n"

        string += "MANAGER DETAILS" + "\n"
        string += "> " + str(self.optimizer.__name__) + "\n"
        string += "> Learning rate: " + str(self.learning_rate) + "\n"
        string += "> Weight decay: " + str(self.weight_decay) + "\n"
        string += self.loss_function.to_string()
        string += "\n"

        string += "NETWORK DETAILS" + "\n"
        string += self.network.to_string()
        string += "\n"

        string += "TRAINING DETAILS" + "\n"
        for i in range(len(self.epoch_stops)):
            string += "epoch {}".format(self.epoch_stops[i]) + "\n"
            if not i == len(self.epoch_stops) - 1:
                string += "> start: {}".format(self.training_starts[i]) + "\n"
                string += "> end: {}".format(self.training_stops[i]) + "\n"

        return string

    def load_from_file(self):
        self.network_converter.load_manager(self)

    @staticmethod
    def from_string(input_string_lines, reader, manager):
        assert(input_string_lines[0] == "DATASET GENERATOR")

        dataset_manager_lines = []
        i = 1
        while not input_string_lines[i] == "":
            dataset_manager_lines.append(input_string_lines[i])
            i += 1
        manager.set_dataset_generator(
            reader.read_generator_lines(dataset_manager_lines)
        )

        i += 2
        manager.set_optimizer(
            reader.get_optimizer(
                input_string_lines[i].replace("> ", "")
            )
        )
        i += 1
        manager.set_learning_rate(
            float(input_string_lines[i].replace("> Learning rate: ", ""))
        )
        i += 1
        manager.set_weight_decay(
            float(input_string_lines[i].replace("> Weight decay: ", ""))
        )

        i += 1
        loss_function_lines = []
        while not input_string_lines[i] == "":
            loss_function_lines.append(input_string_lines[i])
            i += 1
        manager.set_loss_function(
            reader.read_loss_function_lines(loss_function_lines)
        )

        i += 2
        network_lines = []
        while not input_string_lines[i] == "":
            network_lines.append(input_string_lines[i])
            i += 1
        manager.set_network(
            reader.read_network_lines(network_lines)
        )

        i += 2
        epoch_stops = []
        training_starts = []
        training_stops = []
        while not (i == len(input_string_lines) - 1):
            epoch_stops.append(
                int(input_string_lines[i].replace("epoch ", ""))
            )
            training_starts.append(
                input_string_lines[i+1].replace("> start: ", "")
            )
            training_stops.append(
                input_string_lines[i+2].replace("> end: ", "")
            )
            i += 3
        epoch_stops.append(
            int(input_string_lines[i].replace("epoch ", ""))
        )
        manager.set_training_starts(training_starts)
        manager.set_training_stops(training_stops)
        manager.set_epoch_stops(epoch_stops)
