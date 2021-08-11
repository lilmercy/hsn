import hsn

# 用户自定义设置
MODEL_NAME = 'histonet_X1.7_clrdecay_5' # 模型名称
INPUT_NAME = '01_tuning_patch' # 输入图像名称
INPUT_MODE = 'patch'                    # {'patch', 'wsi'} 输入图像类别
INPUT_SIZE = [224, 224]                 # [<int>, <int>] > 0 输入图像大小
HTT_MODE = 'both'                       # {'both', 'morph', 'func', 'glas'} HTT（Histological Tissue Type, 组织学类型）
BATCH_SIZE = 16                         # int > 0 批量大小
GT_MODE = 'on'                          # {'on', 'off'} 是否根据ground-truth注释评估分割结
RUN_LEVEL = 3                           # {1: HTT confidence scores, 2: Grad-CAMs, 3: Segmentation masks} 在 HistoSegNet 运行到哪个阶段
SAVE_TYPES = [1, 1, 1, 1]               # {HTT confidence scores, Grad-CAMs, Segmentation masks, Summary images} 要保存用于调试的类型
VERBOSITY = 'NORMAL'                    # {'NORMAL', 'QUIET'} 调试信息详细程度
DOWNSAMPLE_FACTOR = 1                   # 下采样

# 设置 HistoSegNetV1
hsn = hsn_v1.HistoSegNetV1(params={'input_name': INPUT_NAME, 'input_size': INPUT_SIZE, 'input_mode': INPUT_MODE,
                                   'down_fac': DOWNSAMPLE_FACTOR, 'batch_size': BATCH_SIZE, 'htt_mode': HTT_MODE,
                                   'gt_mode': GT_MODE, 'run_level': RUN_LEVEL, 'save_types': SAVE_TYPES,
                                   'verbosity': VERBOSITY})

# 查找图像
hsn.find_img()
hsn.analyze_img()

# 加载 HistoNet
hsn.load_histonet(params={'model_name': MODEL_NAME})

# 批量操作
hsn.run_batch()
