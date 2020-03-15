from composed_layer.auto_encoder import ComposedAutoEncoder
from dataset.primitives import PrimitiveShapes

RADIUS1 = 0.3
RADIUS2 = 1.0

dataset = PrimitiveShapes.generate_dataset(1, 2000)
points = dataset[0].pos
print(points.size())
print()

encoder = ComposedAutoEncoder()
out = encoder(points)

print()
print(out.size())












