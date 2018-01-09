# SuperResolution-Chainer
Super Resolution by Chainer(v3) and python3.  
I used a [DenseNet](https://arxiv.org/abs/1608.06993)-based Generator and [SNGAN](https://drive.google.com/file/d/0B8HZ50DPgR3eSVV6YlF3XzQxSjQ/view) by [Chainer](https://github.com/pfnet-research/chainer-gan-lib) to train this model.   
This repository just provides the generator model, however, I have used GAN to train it actually.  
If you have any question, please feel free to contact me.

## Usage

### compare   
```
python compare_image --input_file/-i  filename
```

It will downsize the given image to the low resolution image with a factor=2, then upsize it by bicubic and SR-method respectively to generate super resolution image with a factor=2 and compare the PSNR/SSIM between the SR image and ground truth.

### generate  
```
python generate_2x --input_file/-i  filename
```
It will generate a 2x SR image of the given image by SR-method. 

## Result

### compare
#### ground truth
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly.png)

#### low resolution
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_low.png)

#### bicubic
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_bic.png)

#### super resolution
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_super.png)

#### generate 2x
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_super_2x.png)


