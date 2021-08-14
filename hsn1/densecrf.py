import os
import numpy as np
#import pydensecrf as dcrf
import Pydensecrf.pydensecrf.densecrf as dcrf  # always import failed with "No module named pydensecrf.desencrf"
from pydensecrf.utils import unary_from_softmax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 定义一个实现dense CRF (全连接条件随机场)的类
class DenseCRF:
    """Class for implementing a dense CRF"""
   
    # 设定一些初始参数
    def __init__(self):
        self.gauss_sxy = 3
        self.gauss_compat = 30
        self.bilat_sxy = 10
        self.bilat_srgb = 20
        self.bilat_compat = 50
        self.n_infer = 5
   
    # 从文件中加载全连接（稠密）条件随机场的配置
    def load_config(self, path):
        """Load dense CRF configurations from file"""
        # 如果存在路径path, 则读写磁盘中path的数据, 并将config[0]设置为初始值
        if os.path.exists(path):
            config = np.load(path)
            self.gauss_sxy, self.gauss_compat, self.bilat_sxy, self.bilat_srgb, self.bilat_compat, self.n_config = config[0]
        else:
            # 如果不存在路径path, 报错
            print('Warning: dense CRF config file ' + path + ' does not exist - using defaults')
    
    # 运行dense CRF(全连接条件随机场), 输入: 概率图和输入图像
    def process(self, probs, images):
        """
        Run dense CRF, given probability map and input image

        Parameters（输入参数）
        ----------
        probs : numpy 4D array -- 代表类别概率映射的一个 numpy 四维数组
            The class probability maps, in batch 
        images : numpy 4D array -- 代表原始输入图像的一个 numpy 四维数组
            The original input images, in batch

        Returns（返回 / 输出）
        -------
        maxconf_crf : numpy 3D array -- 来自 dense CRF 的离散类别分割映射 三维 numpy 数组
            The discrete class segmentation map from dense CRF, in batch
        crf : numpy 4D array -- 来自 dense CRF 的连续类别概率映射 四维 numpy 数组
            The continuous class probability map from dense CRF, in batch
        """

        # 设置可变参数（尺寸大小）
        # probs（类别概率映射）四维数组矩阵第一维度的长度（行数）设置为 num_input_images（输入图像数量）
        num_input_images = probs.shape[0]
        # images（原始输入图像）四维数组矩阵第二维度的长度（列数）设置为 num_classes（类别数量）
        num_classes = probs.shape[1]
        # 从输入图像的四维数组的第 2 到第 4 数组值 设置为size
        size = images.shape[1:3]
        # 
        crf = np.zeros((num_input_images, num_classes, size[0], size[1]))
        for iter_input_image in range(num_input_images):
            pass_class_inds = np.where(np.sum(np.sum(probs[iter_input_image], axis=1), axis=1) > 0)
            # Set up dense CRF 2D
            d = dcrf.DenseCRF2D(size[1], size[0], len(pass_class_inds[0]))
            cur_probs = probs[iter_input_image, pass_class_inds[0]]
            # Unary energy
            U = np.ascontiguousarray(unary_from_softmax(cur_probs))
            d.setUnaryEnergy(U)
            # Penalize small, isolated segments
            # (sxy are PosXStd, PosYStd)
            d.addPairwiseGaussian(sxy=self.gauss_sxy, compat=self.gauss_compat)
            # Incorporate local colour-dependent features
            # (sxy are Bi_X_Std and Bi_Y_Std,
            #  srgb are Bi_R_Std, Bi_G_Std, Bi_B_Std)
            d.addPairwiseBilateral(sxy=self.bilat_sxy, srgb=self.bilat_srgb, rgbim=np.uint8(images[iter_input_image]),
                                   compat=self.bilat_compat)
            # Do inference
            Q = d.inference(self.n_infer)
            crf[iter_input_image, pass_class_inds] = np.array(Q).reshape((len(pass_class_inds[0]), size[0], size[1]))
        maxconf_crf = np.argmax(crf, axis=1)
        return maxconf_crf, crf
