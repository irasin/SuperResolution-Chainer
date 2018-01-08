############################################################################################

# This file is used to compare bicubic and my Super Resolution method by compute PSNR/SSIM of the given image.

# Author: CHEN CHEN

#############################################################################################


import numpy as np
import cv2
import argparse
from chainer import Variable, serializers
from skimage.measure import compare_ssim, compare_psnr, compare_mse
from model import Generator


def compare(image1, image2):
    mse = compare_mse(image1, image2)
    psnr = compare_psnr(image1, image2)
    ssim = compare_ssim(image1, image2, multichannel=True)
    print("mse:{:.4f}\npsnr:{:.4f}\nssim:{:.4f}".format(mse, psnr, ssim))


class GeneralGen:
    def __init__(self, image_path, gen, gen_data, cropsize=128):
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
        """get image of (128*n, 128*n, 3) by zero padding and change image into (n_H, n_W, cropsize, cropsize, C)"""
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

    def img2lrbic(self, image):
        """return low resolution image  and bicubic image"""
        image_low = cv2.resize(image,
                               (int(self.cropsize / 2), int(self.cropsize / 2)),
                               interpolation=cv2.INTER_CUBIC)
        image_bic = cv2.resize(image_low,
                               (self.cropsize, self.cropsize),
                               interpolation=cv2.INTER_CUBIC)
        return image_low, image_bic

    def tochainer(self, cv2_image):
        """ndarray --> Variable"""
        cv2_image = cv2_image.astype(np.float32).transpose((2, 0, 1)) / 255.0
        C, H, W = cv2_image.shape
        cv2_image = cv2_image.reshape((1, C, H, W))
        return Variable(cv2_image)

    def gensuper(self, image):
        """low resolution --> super resolution"""
        image = self.tochainer(image)
        gen = self.gen
        sr = gen(image)
        sr = np.asarray(np.clip(sr.data * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = sr.shape
        sr = sr.reshape((3, H, W))
        sr = sr.transpose(1, 2, 0)
        return sr

    def mergeimage(self, image_list, if_low=False):
        """merge imagelist to imagearry and delete parts of zeros padding"""
        H, W, _ = image_list[0].shape
        image_array = np.asarray(image_list).reshape(self.n_H, self.n_W, H, W, self.C)
        image_array = image_array.transpose(0, 2, 1, 3, 4)
        image_array = image_array.reshape(self.n_H * H, self.n_W * W, self.C)
        if if_low:
            image_array = image_array[:self.H//2, :self.W//2, :]
        else:
            image_array = image_array[:self.H, :self.W, :]
        return image_array

    def processing(self):
        """main process to get image_low, image_bic, image_super"""
        serializers.load_npz(self.gen_data, self.gen)
        res = self.zeropadimage()
        image_low = []
        image_super = []
        for i in range(res.shape[0]):
            h = res[i]
            for j in range(h.shape[0]):
                w = h[j]
                lr, bic = self.img2lrbic(w)
                sr = self.gensuper(lr)
                image_low.append(lr)
                image_super.append(sr)
        image_low = self.mergeimage(image_low, if_low=True)
        image_bic = cv2.resize(image_low, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        image_super = self.mergeimage(image_super)
        return image_low, image_bic, image_super


def generalgen(image_path, gen_data):
    gen = Generator()
    image_low, image_bic, image_super = GeneralGen(image_path, gen, gen_data).processing()
    save_path, _ = image_path.split(".")
    cv2.imwrite(save_path + "_low.png", image_low)
    cv2.imwrite(save_path + "_bic.png", image_bic)
    cv2.imwrite(save_path + "_super.png", image_super)


def main():
    parser = argparse.ArgumentParser(description='Super Resolution method by Chainer')
    parser.add_argument('--input_file', '-i', type=str, default="butterfly.png",
                        help='the input file to SR')
    parser.add_argument('--gen_data', '-gd', type=str, default="gen_data.npz",
                        help='the data for generator to load')
    args = parser.parse_args()

    generalgen(args.input_file, args.gen_data)

    image_high = cv2.imread(args.input_file)
    save_path, _ = args.input_file.split(".")
    image_bic = cv2.imread(save_path + "_bic.png")
    image_super = cv2.imread(save_path + "_super.png")

    compare(image_high, image_bic)
    compare(image_high, image_super)


if __name__ == '__main__':
    main()
