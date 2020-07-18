from torch_geometric.data import DataLoader


class DatasetGenerator:

    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate_loaders(self):
        train_loader = DataLoader(
            dataset=self.generate_train_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        val_loader = DataLoader(
            dataset=self.generate_validation_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        return train_loader, val_loader

    def generate_train_dataset(self):
        pass

    def generate_validation_dataset(self):
        pass

    def get_sizes(self):
        return self.get_train_size(), self.get_val_size()

    def get_train_size(self):
        pass

    def get_val_size(self):
        pass

    def to_string(self):
        string = "> batch size: " + str(self.batch_size) + "\n"
        string += "> shuffle: " + str(self.shuffle) + "\n"

        return string

    @staticmethod
    def from_string(input_lines):
        batch_size = int(input_lines[0].replace("> batch size: ", ""))
        shuffle = input_lines[1].replace("> shuffle: ", "") == "True"

        return batch_size, shuffle
