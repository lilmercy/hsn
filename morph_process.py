import os
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

# set the path
# current path
cur_path = os.path.abspath(os.path.curdir)
# origin_path
img_path = os.path.join(cur_path, 'img')
patch_path = os.path.join(img_path, '01_tuning_patch')
origin_path = os.path.join(patch_path, '132_R02.png_crop_7.png')
# out path
out_path = os.path.join(cur_path, 'out', '01_tuning_patch', 'morph')
# process
# ablative adjust path
ablative_adjust =os.path.join(out_path, 'ablative_Adjust', '132_R02.png_crop_7.png')
# ablative_CRF path
ablative_CRF = os.path.join(out_path, 'ablative_CRF', '132_R02.png_crop_7.png')
# ablative_GradCAM path
ablative_GradCAM = os.path.join(out_path, 'ablative_GradCAM', '132_R02.png_crop_7.png')
# overlay path
overlay = os.path.join(out_path, 'overlay', '132_R02.png_crop_7.png')
# segmentation result
morph_seg_path = os.path.join(out_path, 'patch', '132_R02.png_crop_7.png')
# ground-truth path
gt_path = os.path.join(cur_path, 'gt')
morph_gt_path = os.path.join(gt_path, '01_tuning_patch', 'morph', '132_R02.png_crop_7.png')
# vertical path
vertical = os.path.join(out_path, 'vertical', '132_R02.png_crop_7.png.png')

# set the image
origin_img = mpimg.imread(origin_path)
ablative_adjust_img = mpimg.imread(ablative_adjust)
ablative_CRF_img = mpimg.imread(ablative_CRF)
ablative_GradCAM_img = mpimg.imread(ablative_GradCAM)
overlay_img = mpimg.imread(overlay)
morph_seg_img = mpimg.imread(morph_seg_path)
morph_gt_img = mpimg.imread(morph_gt_path)
vertical_img = mpimg.imread(vertical)

# the origin image
plt.imshow(origin_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the origin image')
plt.show()

# the ablative_adjust image
plt.imshow(ablative_adjust_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the ablative_adjust image')
plt.show()

# the ablative_CRF image
plt.imshow(ablative_CRF_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the ablative_CRF image')
plt.show()

# the ablative_GradCAM image
plt.imshow(ablative_GradCAM_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the ablative_GradCAM image')
plt.show()

# the overlay image
plt.imshow(overlay_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the overlay image')
plt.show()

# the morph segmentation
plt.imshow(morph_seg_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the morph segmentation')
plt.show()

# the morph ground-truth
plt.imshow(morph_gt_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the morph ground-truth')
plt.show()

# the vertical
plt.imshow(vertical_img) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.title('the vertical')
plt.show()