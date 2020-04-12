import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.encoder import SimpleRelativeEncoder
from relative_layer.decoder import SimpleRelativeDecoder


class SingleLayerNetwork(nn.Module):
    """
    This module represents a single layer of the full pointcloud autoencoder network. It is
        used to train on local neighborhoods in different layers of the network. They are
        only trained to encode this local neighborhood and no other information that can
        be present at points in the full network.
    """
    def __init__(self, nb_layers, nbs_neighbors, final_layer, radius):
        """
        Initializes all parameters of the SingleLayerNetwork. The encoder and decoder are
            as default set to SimpleRelativeEncoder and SimpleRelativeDecoder. These can
            be changed through the set_encoder() and set_decoder() functions.
        :param nb_layers: The layer number in the full network that this network represents.
        :param nbs_neighbors: A list of integers which represents the number of neighbors in
            each layer. These numbers are used to calculate the ratio for the fps sampling.
        :param final_layer: Boolean representing whether or not this network represents the
            final layer in a full network. In the final layer, the origin is used as midpoint.
        :param radius: The radius used in the layer that this network represents. The radius is
            used for normalization of the input points.
        """
        super(SingleLayerNetwork, self).__init__()

        assert(nb_layers == len(nbs_neighbors))

        self.encoder = SimpleRelativeEncoder()
        self.decoder = SimpleRelativeDecoder()

        self.nb_layers = nb_layers
        self.nbs_neighbors = nbs_neighbors
        self.final_layer = final_layer

        self.radius = radius

        self.normal_required = False

    def forward(self, batch_object):
        """
        The train function of this network.
        :param batch_object: Batch object of the pytorch_geometric librarie. The object should
            have a pos attribute and a batch attribute. A norm attribtue should be provided when
            normal_required is set to true.
        :return: The origal points and the cluster they belong to is returned as well as the
            output points and the cluster that they belong to.
        """
        # points = (nb_batch * nb_points) x 3
        # points = (nb_batch * nb_points) x 1
        points = batch_object.pos
        batch = batch_object.batch

        # normals = (nb_batch * nb_points) x 3
        if self.normal_required:
            normals = batch_object.norm

        current_points = points
        current_fps_points = points
        current_batch = batch
        current_fps_batch = batch
        if self.normal_required:
            current_normals = normals
            current_fps_normals = normals

        if not self.final_layer:
            nb_loops = self.nb_layers
        else:
            nb_loops = self.nb_layers - 1

        fps_inds_list = []
        for i in range(nb_loops):
            current_points = current_fps_points
            current_batch = current_fps_batch

            ratio = 1/self.nbs_neighbors[i]
            fps_inds = gnn.fps(current_points, ratio=ratio, batch=current_batch)
            fps_inds_list.append(fps_inds)

            current_fps_points = current_points[fps_inds]
            current_fps_batch = current_batch[fps_inds]
            if self.normal_required:
                current_normals = current_fps_normals
                current_fps_normals = current_normals[fps_inds]

        if not self.final_layer:
            rad_cluster, rad_inds = gnn.radius(
                current_points, current_fps_points,
                batch_x=current_batch, batch_y=current_fps_batch, r=self.radius
            )

            rad_points = current_points[rad_inds]
            mid_points = current_fps_points[rad_cluster]

            original_points = rad_points
            relative_points = (rad_points - mid_points) / self.radius
            cluster = rad_cluster

            if self.normal_required:
                normals_in = current_normals[rad_inds]

        else:
            original_points = current_fps_points
            relative_points = current_fps_points / self.radius
            cluster = current_fps_batch
            if self.normal_required:
                normals_in = current_fps_normals

        if not self.normal_required:
            encoded = self.encoder(relative_points, cluster)
        else:
            encoded = self.encoder(relative_points, normals_in, cluster)
        decoded, cluster_decoded = self.decoder(encoded)

        if not self.final_layer:
            out = (decoded * self.radius) + current_fps_points[cluster_decoded]
        else:
            out = decoded * self.radius

        return original_points, cluster, out, cluster_decoded

    def get_number_neighbors(self):
        """
        Get the number of neighbors of the layer that this network represents. This is the
            last element in the list of numbers of neighbors.
        :return: Integer value that is the number of neighbors of the layer that this network
            represents.
        """
        return self.nbs_neighbors[-1]

    def set_encoder(self, encoder):
        """
        Set the encoder of this network. The encoder will be notified of this event and should
            change the normal requirement to its needs.
        :param encoder: The new encoder for this network.
        """
        print("WARNING: all training results so far will be overwritten and lost! [SingleLayerNetwork]")
        self.encoder = encoder
        encoder.accept_parent(self)

    def set_decoder(self, decoder):
        """
        Set the decoder of this network. The decoder will be notified of this event and should
            make sure the number of neighbors of both networks agree.
        :param decoder: The new decoder for this network.
        :return:
        """
        print("WARNING: all training results so far will be overwritten and lost! [SingleLayerNetwork]")
        self.decoder = decoder
        decoder.accept_parent(self)

    def normal_required(self):
        """
        :return: Return whether or not the encoder of this network requires information about normals.
        """
        return self.normal_required()

    def set_normal_required(self):
        """
        The encoder will receive normal information during trainng.
        """
        self.normal_required = True

    def set_normal_not_required(self):
        """
        The encoder will not receive normal information during training.
        """
        self.normal_required = False

