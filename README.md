# Generative Mesh Models
In this project, deep generative neural networks are used to create unseen point clouds.
The project uses neighborhoods of points to encoded and decoded point clouds in a layered structure.

## Encoding
In each layer, Furthest Point Sampling (FPS) is used to select points that best represent the point cloud in that layer. Next, the neighborhoods are defined by searching all points within a certain radius of the fps points. The neighborhoods are normalized, using the fps points and the radius, so that parameter sharing can be applied.

![Encoder](/images/encoder.png)


## Decoding
In each layer, the decoder outputs a set of relative points. Using the points of the previous layer and the radius of current layer, these relative points are denormalized. The denormalized points of the final layer form the orignal point cloud.

![Decoder](/images/decoder.png)

## Result
The image below show the result of the network for some basic geometric shapes. Currently, the network is being tested on data from ShapeNet.

![Result1](/images/full_network_result.PNG)

Due to the layered sturcture of the network, the different stages of the network learning can be visualized. Currently, the network learns differently than expected. Experiments are currently being conducted where the loss funtion is adapted to force the network to better represent the intermediate point clouds as well.

![Result2](/images/stages_torus.PNG)



