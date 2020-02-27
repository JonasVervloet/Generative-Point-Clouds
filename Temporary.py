import numpy as np


def soft_threshold(lamb, weight):
    if abs(weight) <= lamb:
        return 0
    elif weight < 0:
        return weight + lamb
    else:
        return weight - lamb


def soft_vectorized(lamb, weight_vector):
    length = len(weight_vector)
    new_vector = np.zeros(length)
    for i in range(length):
        new_vector[i] = soft_threshold(lamb, weight_vector[i])
    return new_vector


def update_weights(weights, x, y, learning_rate, lamb):
    update = np.transpose(x).dot(y - x.dot(weights))
    inp = weights + learning_rate * update
    return soft_vectorized(lamb, inp)


w0 = np.array([0.0, 0.5])
lr = 0.01
la = 0.2

x = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.4, 1.5, 2.3, 2.9]])
x = np.transpose(x)

y = np.array([1.3, 2.8, 4.3, 5.3, 5.3])

wi = w0
print(w0)
for i in range(1, 10):
    wi = update_weights(wi, x, y, lr, la)
    print(wi)
