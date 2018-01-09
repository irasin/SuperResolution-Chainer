############################################################################################

# This file is used to generate a 2x super resolution image of the the given image.

# Author: CHEN CHEN

#############################################################################################

import numpy as np
import cv2
import argparse
import chainer
from chainer import Variable, serializers
from model import Generator


class Generate2X:
    """split the given image into 64*64*3 images and generate 2x super resolution 128*128*3 images,
       finally merge them into a image"""
    def __init__(self, image_path, gen, gen_data, cropsize=64):
        self.image = cv2.imread(image_path)
        self.gen = gen
        self.gen_data = gen_data
        self.cropsize = cropsize
        self.H, self.W, self.C = self.image.shape
        self.Hzeropad = cropsize - self.H % cropsize
        self.Wzeropad = cropsize - self.W % cropsize
        self.n_H = None
        self.n_W = None

    def zeropadimage(self):
        """generate image of (64*n, 64*n, 3) by zero padding
        and change image into (n_H, n_W, cropsize, cropsize, C)"""
        Wzeropad = np.zeros((self.H, self.Wzeropad, self.C))
        res = np.hstack((self.image, Wzeropad))
        Hzeropad = np.zeros((self.Hzeropad, self.W + self.Wzeropad, self.C))
        res = np.vstack((res, Hzeropad))
        H, W, _ = res.shape
        self.n_H = H // self.cropsize
        self.n_W = W // self.cropsize
        res = res.reshape(self.n_H,  self.cropsize, self.n_W, self.cropsize, self.C)
        res = res.transpose(0, 2, 1, 3, 4)
        return res

    def tochainer(self, cv2_image):
        """array --> Variable"""
        cv2_image = cv2_image.astype(np.float32).transpose((2, 0, 1)) / 255.0
        C, H, W = cv2_image.shape
        cv2_image = cv2_image.reshape((1, C, H, W))
        return Variable(cv2_image)

    def gensuper(self, image):
        """low resolution --> super resolution"""
        gen = self.gen
        image = self.tochainer(image)
        with chainer.using_config('train', False):
            sr = gen(image)
        sr = np.asarray(np.clip(sr.data * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = sr.shape
        sr = sr.reshape((3, H, W))
        sr = sr.transpose(1, 2, 0)
        return sr

    def mergeimage(self, image_list):
        """merge imagelist to imagearry and delete parts of zeros padding"""
        H, W, _ = image_list[0].shape
        image_array = np.asarray(image_list).reshape(self.n_H, self.n_W, H, W, self.C)
        image_array = image_array.transpose(0, 2, 1, 3, 4)
        image_array = image_array.reshape(self.n_H * H, self.n_W * W, self.C)
        image_array = image_array[:2*self.H, :2*self.W, :]
        return image_array

    def processing(self):
        """main process to get image_super"""
        res = self.zeropadimage()
        serializers.load_npz(self.gen_data, self.gen)
        image_super = []
        for i in range(res.shape[0]):
            h = res[i]
            for j in range(h.shape[0]):
                w = h[j]
                sr = self.gensuper(w)
                image_super.append(sr)
        return self.mergeimage(image_super)


def generate_2x(image_path, gen_data):
    gen = Generator()
    image_super = Generate2X(image_path, gen, gen_data).processing()
    save_path, _ = image_path.split(".")
    cv2.imwrite(save_path + "_super_2x.png", image_super)


def main():
    parser = argparse.ArgumentParser(description='Super Resolution method by Chainer')
    parser.add_argument('--input_file', '-i', type=str, default="butterfly.png",
                        help='the input file to SR')
    parser.add_argument('--gen_data', '-gd', type=str, default="gen_data.npz",
                        help='the data for generator to load')
    args = parser.parse_args()

    generate_2x(args.input_file, args.gen_data)


if __name__ == '__main__':
    main()
