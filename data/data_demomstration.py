import numpy as np
'''
data_demonstration 用来读取并可视化 hsn/data 文件夹下的数据
func_optimal_pcc.npy
morph_optimal_pcc.npy
glas_optimal_pcc.npy
histonet_X1.7_clrdecay_5.h5 -- histonet 的网络结构
histonet_X1.7_clrdecay_5.mat
histonet_X1.7_clrdecay_5.json -- histonet
'''
# read the 'func_optimal_pcc.npy'
func_npy = np.load('/home/lmx/hsn/data/func_optimal_pcc.npy')
print("-------- func type --------")
print(type(func_npy))
print("-------- func shape --------")
print(func_npy.shape)
print("-------- func data --------")
print(func_npy)

# morph_optimal_pcc.npy
morph_npy = np.load('/home/lmx/hsn/data/morph_optimal_pcc.npy')
print("-------- morph type --------")
print(type(morph_npy))
print("-------- morph shape --------")
print(morph_npy.shape)
print("-------- morph data --------")
print(morph_npy)

# glas_optimal_pcc.npy
glas_npy = np.load('/home/lmx/hsn/data/glas_optimal_pcc.npy')
print("------- glas type --------")
print(type(glas_npy))
print("-------- glas shape --------")
print(glas_npy.shape)
print("-------- glas data --------")
print(glas_npy)

# histonet_X1.7_clrdecay_5.h5
import h5py
with h5py.File('/home/lmx/hsn/data/histonet_X1.7_clrdecay_5.h5', "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)

# histonet_X1.7_clrdecay_5.mat
import scipy.io as scio
dataFile = '/home/lmx/hsn/data/histonet_X1.7_clrdecay_5.mat'
mat_data = scio.loadmat(dataFile)
print(type(mat_data))
print(mat_data)

# histonet_X1.7_clrdecay_5.json
import json
with open('/home/lmx/hsn/data/histonet_X1.7_clrdecay_5.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    print('这是文件中的json数据：',json_data)
    print('这是读取到文件数据的数据类型：', type(json_data))