# HistoSegNet (V1)

### Original author: Lyndon Chan -- http://lyndonchan.github.io/

### Original repo: https://github.com/lyndonchan/hsn_v1.git

### Paper: https://sci-hub.mksa.top/10.1109/iccv.2019.01076

# Requirement (I used)

+ I used Pycharm (Professional) and ssh sever
+ `python==3.6` (pycharm do not support python==3.5 intepreter)
+ `keras==2.2.4`
+ `tensorflow==1.13.1`
+ `numpy==1.16.2`
+ `cython==0.29.24`
+ `pydensecrf==1.0rc3`
+ `opencv-python-headless==4.5.3.56`
+ `scikit-image==0.14.2`
+ `scipy==1.2.0`
+ `pandas`
+ `matplotlib`

# Setup (For reference only)

+ Create a conda environment for HistoSegNet  `conda create -n histo python==3.6`

+ Activate the conda environment  `conda activate histo`

+ Install packages

  `pip install -i https://pypi.douban.com/simple/ keras==2.2.4`

  `pip install -i https://pypi.douban.com/simple/ tensorflow==1.13.1`

  `pip install -i https://pypi.douban.com/simple/ numpy==1.16.2`

  `pip install -i https://pypi.douban.com/simple/ cython`

  `conda install -c conda-forge pydensecrf=1.0rc3`

  `pip install -i https://pypi.douban.com/simple/ opencv-python-headless==4.5.3.56`

  `pip install -i https://pypi.douban.com/simple/ scipy==1.2.0`

  `pip install -i https://pypi.douban.com/simple/ scikit-image==0.14.2 `

  `pip install -i https://pypi.douban.com/simple/ pandas`

  `pip install -i https://pypi.douban.com/simple/ matplotlib`

+ Run the demo

  `cd ./your code folder directory/`

  `python demo_01_segment_patches` for run the demo of segment patches

  `python demo_02_glas_patches`for run the demo glas patches

+ The segement result are in the 'out' folder

# Description 

## ./data/

The files about classifacation CNN histonet.

### data_demonstration.py

Run this and you can get the visualization result of all the files in /data/.

### func / morph / glas_optimal_pcc.npy

### histonet_X1.7_clrdecay_5.h5 / json / mat

The model & weights files of classification CNN histonet.



## ./gt/

The ground-truth of tuning set (patch-level) and glas set (full / slide - level & patch level). 

### ./01_tuning_patch/

+ **./func/** 

  The functional ground-truth of patch-level tuning set.

+ **./morph/**

  The morphology ground-truth of patch-level tuning set.

+ **gt_labels.csv** 

  The ground-truth label of patch-level tuning set.

### ./02_glas_full/

+ **./glas/**

  The full / slide - level ground-truth of glas set.

+ **./glas_multi_gland/** 

  The multi gland .bmp ground-truth of glas set.

+ **./glas_single_gland/** 

  The single gland .bmp ground-truth of glas set.

### ./02_glas_patch/glas/

The ground-truth of patch-level glas set.



## ./img/

The need-to-segment images of tuning set (patch-level) and glas set (full / slide - level & patch level). 

### ./01_tuning_patch/

The patch-level need-to-segment images of tuning set.

### ./02_glas_full/ 

The full (slide) - level need-to-segment images of glas set.

### ./02_glas_patch/glas/ 

The patch-level need-to-segment images of glas set.



## ./hsn1/

### adp.py 

About atlas of Digital Pathology

### densecrf.py 

About the dense conditional random field (dense CRF)

### gradcam.py 

About Gradient-weighted Class Activation Mapping (Grad-CAM)

### histonet.py 

About the classification CNN (histonet)

### hsn_v1.py

About the segmentation network HistoSegNet (V1)



### demo_01_demonstration.py

Run demo_01_demonstration.py, you will get the visualization result of demo_01_segment_patches.py

### demo_02_demonstration.py

Run demo_02_demonstration.py, you will get the visualization result of demo_02_segment_glas_patches.py

### demo_01_segment_patches.py

The demo of segmentation of tuning set.

### demo_02_segment_glas_patches.py

The demo of segmentation of glas set.

### morph_process.py

The visualization of the morph segmentation of tuning set.



### I will continue to improve this repo to better implement it, because this is a fabulous work for me.

### If you need any help, please email me by lmxhrb@163.com.

### Sincere respect and thanks to Lyndon Chan！
