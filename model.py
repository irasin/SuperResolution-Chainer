import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from common.sn.sn_convolution_2d import SNConvolution2D


class DenseLayer(chainer.Chain):
    def __init__(self, in_channel, growth_rate, bn_size, if_gen=False):
        super(DenseLayer, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            if if_gen:
                self.conv1 = L.Convolution2D(None, bn_size * growth_rate, 1, 1, 0, initialW=initialW)
                self.conv2 = L.Convolution2D(None, growth_rate, 3, 1, 1, initialW=initialW)
            else:
                self.conv1 = SNConvolution2D(None, bn_size * growth_rate, 1, 1, 0, initialW=initialW)
                self.conv2 = SNConvolution2D(None, growth_rate, 3, 1, 1, initialW=initialW)
            self.in_ch = in_channel
            self.if_gen = if_gen

    def __call__(self, x):
        if self.if_gen:
            h = F.relu(self.conv1(x))
            h = self.conv2(h)
        else:
            h = F.leaky_relu(self.conv1(x))
            h = self.conv2(h)
        return F.concat((x, h))


class DenseBlock(chainer.Chain):
    def __init__(self, n_layers, in_channel, bn_size, growth_rate, if_gen=False):
        super(DenseBlock, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                tmp_in_channel = in_channel + i * growth_rate
                setattr(self, 'denselayer{}'.format(i + 1), DenseLayer(tmp_in_channel, growth_rate, bn_size, if_gen))
        self.n_layers = n_layers
        self.if_gen = if_gen

    def __call__(self, x):
        h = x
        for i in range(1, self.n_layers + 1):
            h = getattr(self, 'denselayer{}'.format(i))(h)
            if self.if_gen:
                h = h
            else:
                h = F.leaky_relu(h)
        return h


class Transition(chainer.Chain):
    def __init__(self, in_channel):
        super(Transition, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.conv1 = SNConvolution2D(in_channel, in_channel, 1, 1, 0, initialW=initialW)
            self.conv2 = SNConvolution2D(in_channel, in_channel, 2, 2, 0, initialW=initialW)

    def __call__(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = self.conv2(h)
        return h


class PixelShuffler(chainer.Chain):
    def __init__(self, r):
        super(PixelShuffler, self).__init__()
        self.r = r

    def __call__(self, x):
        batch_size, in_channel, height, width = x.shape
        out_channel = int(in_channel / (self.r * self.r))
        assert out_channel * self.r * self.r == in_channel
        h = x.reshape((batch_size, self.r, self.r, out_channel, height, width))
        h = h.transpose((0, 3, 4, 1, 5, 2))
        return h.reshape((batch_size, out_channel, self.r * height, self.r * width))


class UpScale(chainer.Chain):
    def __init__(self, r, in_channel=64, out_channel=256):
        self.r = r
        self.in_channel = in_channel
        self.out_channel = out_channel
        super(UpScale, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(self.in_channel, self.out_channel, 3, 1, 1)
            self.pixel_shuffler = PixelShuffler(self.r)

    def __call__(self, x):
        return F.relu(self.pixel_shuffler(self.conv(x)))


class Generator(chainer.Chain):
    def __init__(self, n_layers=(4, 4, 4, 4), init_features=64, bn_size=4, growth_rate=12):
        super(Generator, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.in_conv = L.Convolution2D(None, init_features, 9, 1, 4, initialW=initialW)
            n_features = init_features
            self.block1 = DenseBlock(n_layers[0], n_features, bn_size, growth_rate, if_gen=True)
            n_features += n_layers[0] * growth_rate + n_features
            self.block2 = DenseBlock(n_layers[1], n_features, bn_size, growth_rate, if_gen=True)
            n_features += n_layers[1] * growth_rate + n_features
            self.block3 = DenseBlock(n_layers[2], n_features, bn_size, growth_rate, if_gen=True)
            n_features += n_layers[2] * growth_rate + n_features
            self.block4 = DenseBlock(n_layers[3], n_features, bn_size, growth_rate, if_gen=True)
            n_features += n_layers[3] * growth_rate + n_features
            self.mid_conv = L.Convolution2D(None, 64, 3, 1, 1, initialW=initialW)
            self.up1 = UpScale(2)
            # self.up2 = UpScale(2)
            self.out_conv = L.Convolution2D(None, 3, 9, 1, 4, initialW=initialW)

    def __call__(self, x):
        h = first = F.relu(self.in_conv(x))
        h = F.concat((self.block1(h), h))
        h = F.concat((self.block2(h), h))
        h = F.concat((self.block3(h), h))
        h = F.concat((self.block4(h), h))
        h = self.mid_conv(h)
        h = h + first
        h = self.up1(h)
        # h = self.up2(h)
        h = self.out_conv(h)
        return h

