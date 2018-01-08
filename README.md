# SuperResolution-Chainer
Super Resolution by Chainer(v3) and python3.  
I used a [DenseNet](https://arxiv.org/abs/1608.06993)-based Generator and [SNGAN](https://drive.google.com/file/d/0B8HZ50DPgR3eSVV6YlF3XzQxSjQ/view) by Chainer to train this model. 
This repository just provides the generator model, acturally i have used GAN to train it.
If you have any question, please feel free to contact me.

## Usage

### compare   
```
python compare_image --input_file/-i  filename
```

It will generate the low_resolution, bicubic, SR-method image of the given image with factor=2 and compare the PSNR/SSIM

### generate  
```
python generate_2x --input_file/-i  filename
```
It will generate a 2x SR image of the given image by SR-method  

## Result

### compare
#### ground_truth
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly.png)

#### low_resolution
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_low.png)

#### bicubic
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_bic.png)

#### super_resolution
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_super.png)

#### generate 2x
![image](https://github.com/irasin/SuperResolution-Chainer/blob/master/result/butterfly_super_2x.png)


