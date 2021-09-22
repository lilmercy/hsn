import os
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

# set the path
cur_path = os.path.abspath(os.path.curdir)
img_path = os.path.join(cur_path, 'img')
gt_path = os.path.join(cur_path, 'gt')
out_path = os.path.join(cur_path, 'out', '01_tuning_patch')
patch_path = os.path.join(img_path, '01_tuning_patch')
origin_path = os.path.join(patch_path, '132_R02.png_crop_7.png')
morph_gt_path = os.path.join(gt_path, '01_tuning_patch', 'morph', '132_R02.png_crop_7.png')
func_gt_path = os.path.join(gt_path, '01_tuning_patch', 'func', '132_R02.png_crop_7.png')
morph_seg_path = os.path.join(out_path, 'morph', 'patch', '132_R02.png_crop_7.png')
func_seg__path = os.path.join(out_path, 'func', 'patch', '132_R02.png_crop_7.png')

# set the image
origin_img = mpimg.imread(origin_path)
morph_gt_img = mpimg.imread(morph_gt_path)
morph_seg_img = mpimg.imread(morph_seg_path)
func_gt_img = mpimg.imread(func_gt_path)
func_seg_img = mpimg.imread(func_seg__path)

# the origin image
plt.imshow(origin_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the origin image')
plt.show()

# the morph ground-truth
plt.imshow(morph_gt_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the morph ground-truth')
plt.show()

# the morph segmentation
plt.imshow(morph_seg_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the morph segmentation')
plt.show()

# the func ground-truth
plt.imshow(func_gt_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the func ground-truth')
plt.show()

# the func segmentation
plt.imshow(func_seg_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the func segmentation')
plt.show()