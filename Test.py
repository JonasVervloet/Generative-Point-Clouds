from torch_geometric.data import DataLoader

from dataset.primitives import PrimitiveShapes
from full_network.full_nework import FullNetwork

dataset = PrimitiveShapes.generate_dataset(1, 2000)
loader = DataLoader(dataset=dataset, batch_size=2)

done = False

for batch_data in loader:
    if done:
        break
    else:
        done = True
    print(batch_data)

    net = FullNetwork()
    out = net(batch_data.pos, batch_data.batch)

    print()
    print(out.size())













