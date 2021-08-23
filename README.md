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
### I will continue to improve this repo to better implement it, because this is a fabulous work for me.
### If you need any help, please email me by lmxhrb@163.com.
### Sincere respect and thanks to Lyndon Chan！
