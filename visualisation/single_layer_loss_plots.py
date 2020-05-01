import numpy as np
import matplotlib.pyplot as plt

# VARIABLES
RESULTH_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
SAVE_NAME = "LossPlots1/"
SAVE_PATH = RESULTH_PATH + SAVE_NAME
NB_EPOCHS = 100
NB_MAPS = 4
LABELS = [
    "latent size: 32",
    "latent size: 16",
    "latent size: 8",
    "latent size: 4"
]

print("LOADING LOSSES")
train_losses = []
val_losses = []
for i in range(1, NB_MAPS + 1):
    path = RESULTH_PATH + "ParameterReduction{}/".format(i)
    train_losses.append(
        np.load(path + "trainloss_epoch{}.npy".format(NB_EPOCHS))
    )
    val_losses.append(
        np.load(path + "valloss_epoch{}.npy".format(NB_EPOCHS))
    )

arange = range(NB_EPOCHS + 1)

print("PLOT TRAIN LOSSES")
plt.clf()
fig, ax = plt.subplots()
i = 0
for train_loss in train_losses:
    ax.plot(arange, train_loss)
    i += 1
plt.legend(LABELS)
plt.title('Single Layer Network: training losses')
plt.yscale('log')
plt.savefig(
    SAVE_PATH + "train_losses.png"
)

print("PLOT VAL LOSSES")
plt.clf()
fig, ax = plt.subplots()
i = 0
for val_loss in val_losses:
    ax.plot(arange, val_loss)
    i += 1
plt.legend(LABELS)
plt.title('Single Layer Network: validation losses')
plt.yscale('log')
plt.savefig(
    SAVE_PATH + "val_losses.png"
)






