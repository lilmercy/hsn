import numpy as np
# 用于定义与数字病理学图谱相关的信息的类
class Atlas:
    """Class for defining information related to the Atlas of Digital Pathology"""

    def __init__(self):
        self.level1 = ['E', 'C', 'H', 'S', 'A', 'M', 'N', 'G', 'T']
        self.level2 = ['E.M', 'E.T', 'E.P', 'C.D', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M', 'S.E', 'S.C', 'S.R', 'A.W',
                       'A.B', 'A.M', 'M.M', 'M.K', 'N.P', 'N.R', 'N.G', 'G.O', 'G.N', 'T']
        self.level3 = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                       'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                       'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W', 'N.G.T',
                       'G.O', 'G.N', 'T']
        self.level4 = ['E', 'E.M', 'E.T', 'E.P', 'C', 'C.D', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.E',
                       'S.C', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R', 'N.G', 'G',
                       'G.O', 'G.N', 'T'] # level1 + level2
        self.level5 = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C', 'C.D',
                       'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C', 'S.M.S', 'S.E',
                       'S.C', 'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R',
                       'N.R.B', 'N.R.A', 'N.G', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W', 'N.G.T',
                       'G', 'G.O', 'G.N', 'T'] #level1 + level2 + level3
       
        # enumerate -- 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        # np.isin(a, b) -- 用于判定a中的元素在b中是否出现过，如果出现过返回True,否则返回False,最终结果为一个形状和a一模一样的数组
        
        # 如果x存在于self.level3中， 则生成一个与a形状相同的数组（数组内容为True & False）
        # 遍历level5, 打印出在level5中有x与之对应的序号（从0开始）
        self.level5_inds_in_level3 = [i for i,x in enumerate(self.level5) if np.isin(x, self.level3)]
        # 打印在level5中， 有x与之对应的内容
        self.level5_in_level3 = [x for i,x in enumerate(self.level5) if np.isin(x, self.level3)]

        # 形态学类别（label）
        self.morph_classes = ['Background', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.T.X',
                             'E.P', 'C.D.I', 'C.D.R', 'C.L', 'C.X', 'H.E', 'H.K', 'H.Y', 'H.X', 'S.M.C',
                             'S.M.S', 'S.E', 'S.C.H', 'S.C.X', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                             'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'N.G.X']
        # 形态学的颜色[R, G, B]
        self.morph_colours = np.array([[255, 255, 255], [0, 0, 128], [0, 128, 0], [255, 165, 0], [255, 192, 203],
                                      [255, 0, 0], [173, 20, 87], [0, 204, 184], [176, 141, 105], [3, 155, 229],
                                      [158, 105, 175], [216, 27, 96], [131, 81, 63], [244, 81, 30], [124, 179, 66],
                                      [142, 36, 255], [230, 124, 115], [240, 147, 0], [204, 25, 165], [121, 85, 72],
                                      [142, 36, 170], [249, 127, 57], [179, 157, 219], [121, 134, 203], [97, 97, 97],
                                      [167, 155, 142], [228, 196, 136], [213, 0, 0], [4, 58, 236], [0, 150, 136],
                                      [228, 196, 65], [239, 108, 0], [74, 21, 209], [148, 0, 0]])
        
        # 功能性类别 (label)
        self.func_classes = ['Background', 'Other', 'G.O', 'G.N', 'G.X', 'T']
        # 功能性颜色[R, G, B]
        self.func_colours = np.array([[255, 255, 255], [3, 155, 229], [0, 0, 128], [0, 128, 0], [255, 165, 0],
                                      [173, 20, 87]])
        
        # glas图像类别 (label)
        self.glas_valid_classes = ['Other', 'G.O']
        # glas颜色[R, G, B]
        self.glas_valid_colours = np.array([[3, 155, 229], [0, 0, 128]])

        # 检查一下上面的设置是否有重复
        # np.unique -- 去除数组中的重复数字，并进行排序之后输出
        # raise Exception -- 触发异常
        # 如果形态学/功能性的 类别（label）/ 颜色在去除重复后与预设长度不等， 则提示有重复
        if len(np.unique(self.morph_classes, axis=0)) != len(self.morph_classes):
            raise Exception('You have duplicate classes for morphological HTTs')
        if len(np.unique(self.func_classes, axis=0)) != len(self.func_classes):
            raise Exception('You have duplicate classes for functional HTTs')
        if len(np.unique(self.morph_colours, axis=0)) != len(self.morph_colours):
            raise Exception('You have duplicate colours for morphological HTTs')
        if len(np.unique(self.func_colours, axis=0)) != len(self.func_colours):
            raise Exception('You have duplicate colours for functional HTTs')

        # 删除未区分的类别
        # 遍历morph_classes & func_classes, 输出对应类别的索引值
        morph_valid_class_inds = [i for i, x in enumerate(self.morph_classes) if '.X' not in x]
        func_valid_class_inds = [i for i, x in enumerate(self.func_classes) if '.X' not in x]
        # 输出不重复的morph_classes / func_classes / morph_colours / func_colours 为 _valid_
        self.morph_valid_classes = [self.morph_classes[i] for i in morph_valid_class_inds]
        self.func_valid_classes = [self.func_classes[i] for i in func_valid_class_inds]
        self.morph_valid_colours = self.morph_colours[morph_valid_class_inds]
        self.func_valid_colours = self.func_colours[func_valid_class_inds]
        # 遍历level5, 去除x与level3、morph_valid_classes / func_valid_classes 的重复, 输出索引值
        self.level3_valid_inds = [i for i, x in enumerate(self.level5) if np.isin(x,self.level3) and
                                  (np.isin(x, self.morph_valid_classes) or np.isin(x, self.func_valid_classes))]
    
    # 将类别的索引转换为类别本身
    def convert_class_inds(self, class_inds_in, classes_in, classes_out):
        """Convert class indices into the classes themselves"""

        classes_out = np.array([classes_out.index(classes_in[x]) for x in class_inds_in])
        return classes_out
