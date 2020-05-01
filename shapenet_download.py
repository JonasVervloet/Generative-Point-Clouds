from torch_geometric.datasets.shapenet import ShapeNet

CATEGORY = "Airplane"
CATEGORIES = [CATEGORY]

DATA_PATH = "/data/leuven/335/vsc33597/Data/" + CATEGORY + "/"
print(DATA_PATH)

# train
print("TRAIN")
path = DATA_PATH + "train/"
print(path)
dataset = ShapeNet(
    root=path, categories=CATEGORIES,
    include_normals=False, split="train"
)

# validation
print("VALIDATION")
path = DATA_PATH + "validation/"
print(path)
dataset = ShapeNet(
    root=path, categories=CATEGORY,
    include_normals=False, split="val"
)

# test
print("TEST")
path = DATA_PATH + "test/"
print(path)
dataset = ShapeNet(
    root=path, categories=CATEGORY,
    include_normals=False, split="test"
)

print("done")
