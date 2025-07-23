# Image Lens Flare Removal Using Adversarial Curve Learning 

This repository provides the official implementation for the following two papers:

**Improving Lens Flare Removal with General Purpose Pipeline and Multiple Light Sources Recovery** [(ICCV 2023)](https://arxiv.org/abs/2308.16460)

**Image Lens Flare Removal Using Adversarial Curve Learning [(TPAMI 2025)]** 

<img src="result-1.png" width="800px"/>

## Dataset
### Training data
Wu el al. training images are provided in [How to Train Neural Networks for Flare Removal](https://github.com/google-research/google-research/tree/master/flare_removal) (Wu et al. ICCV 2021)

Flare7K training images are provided in [Flare7K: A Phenomenological Nighttime Flare Removal Dataset](https://github.com/ykdai/Flare7K) (Dai et al. NeurIPS 2022). 

Please follow their instructions to access the data.
### Test data
The consumer electronics dataset can be downloaded in [Google Cloud](https://drive.google.com/drive/folders/1J1fw1BggOP-L1zxF7NV0pYhvuZQsmiWY?usp=sharing)
or [Baidu Netdisk](https://pan.baidu.com/s/1KPjDAvNDaLwdxBgtRGPe2A) (password：ptb3).
### Pre-trained Model
The inference code based on Uformer can be downloaded in [Google Cloud](https://drive.google.com/drive/folders/1ngjUh6UzA99-XLi6esK9OdP7ORhU6i8R?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1vrUAS-fIP7Y2aPGjvipU6g) (password: vmxl).

## Code
The backbone of our code is from Wu et al. [How to Train Neural Networks for Flare Removal](https://github.com/google-research/google-research/tree/master/flare_removal). We imporve the synthesis pipeline and light source recovery method to obtain better performance on flare removal. 
### Train
```
python train.py	  --flarec_dir=path/to/captured/flare   --flares_dir=path/to/simulated/flare    --scene_dir=path/to/scene/image
```
### Test
```
python  remove_flare.py   --input_dir=path/to/test/image/dir   --out_dir=path/to/output/dir --model=Uformer    --batch_size=2    --ckpt=path/to/pretrained/model
```
