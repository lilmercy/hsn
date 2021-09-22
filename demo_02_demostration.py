import os
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

# set the path
cur_path = os.path.abspath(os.path.curdir)
img_path = os.path.join(cur_path, 'img')
gt_path = os.path.join(cur_path, 'gt', '02_glas_full')
out_path = os.path.join(cur_path, 'out', '02_glas_full')
patch_path = os.path.join(img_path, '02_glas_full')
origin_path = os.path.join(patch_path, 'train_60.png')
gt_path = os.path.join(gt_path, 'glas', 'train_60.png')
seg_path = os.path.join(out_path, 'glas', 'patch', 'train_60.png')


# set the image
origin_img = mpimg.imread(origin_path)
gt_img = mpimg.imread(gt_path)
seg_img = mpimg.imread(seg_path)

# the origin image
plt.imshow(origin_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the origin image')
plt.show()

# the glas ground-truth
plt.imshow(gt_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the glas ground-truth')
plt.show()

# the glas segmentation
plt.imshow(seg_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the glas segmentation')
plt.show()
