import numpy as np
import pandas as pd

##改写成jupyter的！
print("================================================================","读取csv文件中的数据，并命名每列名字:")
data = pd.read_csv("./data/LogiReg_data.txt",header=None,names=["数学","语文","目标值"])
print("原始数据,添加每列字段解释之后:",data.head(5))

print("======================================================================","原始数据转化成矩阵:")
orig_data = data.values

print("=======================================================================","获取指定值对应的样本:")
positive = data[data["目标值"]==1]
print("positive：",positive.head(10))
print("根据字段描述打印具体的一列",data["目标值"].head())

print("================================================================================","样本切片:")
print("data.shape:",data.shape)
print("orig.data.shape:",orig_data.shape)
cols = data.shape[1]
print("获取data的列数:",cols)
X = orig_data[:,0:cols-1]
print("获取第一列到倒数第第一列,左闭右开:",X)
print("获取0-5行:",orig_data[:5])
print("获取0-5行:",orig_data[0:5])
print("获取0-5行,第1列:",orig_data[0:5,1])
print("矩阵的行数，样本总数:",len(orig_data))

print("===========================================================================修改形状 resize reshaep")
print("修改形状返回新矩阵:",orig_data.reshape([100,3]))
print("修改形状并:",orig_data.reshape([100,3]))


print("===============================================================================","np创建zeros矩阵:")
theta = np.zeros([1,3])
print("theta:",theta)
print("矩阵转置theta.T:",theta.T)


