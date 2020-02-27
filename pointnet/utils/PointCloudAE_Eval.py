import torch
from PointCloudAE import SimplePointCloudAE
from torch_geometric.datasets import ShapeNet
from dataset.functionality import ShapeNetFunctionality as snf
import meshplot as mp
from random import sample

RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"


def eval_model(model, name, nb_epochs, test):
    mp.offline()
    pos = test.pos
    np_pos = pos.numpy()
    plot = mp.subplot(
        np_pos, c=np_pos[:, 0], s=[nb_epochs + 1, 1, 0], shading={"point_size": 0.05}
    )
    for epoch in range(nb_epochs):
        model.load_state_dict(
            torch.load(RESULT_PATH + name + "/" + "model_batchsize1_epoch{}.pt".format(epoch))
        )
        model.eval()
        result = model(pos).detach().numpy()
        print(result)
        mp.subplot(
            result, c=result[:, 0], s=[nb_epochs + 1, 1, epoch + 1],
            data=plot, shading={"point_size": 0.05}
        )
    plot.save(RESULT_PATH + name + "/" + "eval")


dataset = ShapeNet(root="C:/Users/vervl/Data/ShapeNet/utils", train=True,
                   categories=["Airplane"])
filtered = snf.filter_data(dataset, 2300)
resampled = snf.simple_resample_to_minimum(filtered)
print("len resampled dataset: {}".format(len(resampled)))
print("size of data in resampled dataset: {}".format(resampled[5].pos.size()))

rnd_test = sample(resampled, 1)[0]
# max_x = max(rnd_test.pos[:, 0])
# max_y = max(rnd_test.pos[:, 1])
# max_z = max(rnd_test.pos[:, 2])
# min_x = min(rnd_test.pos[:, 0])
# min_y = min(rnd_test.pos[:, 1])
# min_z = min(rnd_test.pos[:, 2])
# print("min_x = {}, max_x = {}".format(min_x, max_x))
# print("min_y = {}, max_y = {}".format(min_y, max_y))
# print("min_z = {}, max_z = {}".format(min_z, max_z))
#
#
# net = PointCoudAERandom()
# result = net(rnd_test.pos)
# pos = result.detach().numpy()
# mp.plot(pos, c=pos[:, 0], filename=RESULT_PATH + "test")
# print("result: {}".format(result))
# max_x = max(result[:, 0])
# max_y = max(result[:, 1])
# max_z = max(result[:, 2])
# min_x = min(result[:, 0])
# min_y = min(result[:, 1])
# min_z = min(result[:, 2])
# print("min_x = {}, max_x = {}".format(min_x, max_x))
# print("min_y = {}, max_y = {}".format(min_y, max_y))
# print("min_z = {}, max_z = {}".format(min_z, max_z))
#
# name = "pointcloudae_random"
# nb_epochs = 10
# eval_model(net, name, nb_epochs, rnd_test)
#
# net = PointCloudAE()
# name = "pointcloudeae_fps"
# nb_epochs = 5
# eval_model(net, name, nb_epochs, rnd_test)

net = SimplePointCloudAE()
name = "simpelAE"
nb_epochs = 20
eval_model(net, name, nb_epochs, rnd_test)

# name = "test"
# hulp_net = Linear(6, 3)
# net = DynamicEdgeConv(hulp_net, 3)
# nb_epochs = 50
# eval_model(net, name, nb_epochs, rnd_test)



