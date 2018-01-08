import numpy as np
from chainer import serializers, datasets
from inception_score import Inception, inception_score

model = Inception()
serializers.load_hdf5('inception_score.model', model)

train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
mean, std = inception_score(model, test)

print('Inception score mean:', mean)
print('Inception score std:', std)
