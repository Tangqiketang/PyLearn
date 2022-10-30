
"""
感知机解决分类问题。
感知机--》神经元--》多个神经元--》神经网络NN   不同结构解决不同问题

neural network,NN,神经网络：
    1.基础NN：单层感知机、线性NN，BPNN，HopfieldNN
    2.进阶NN：波尔兹曼机，受限波尔兹曼机，递归NN
    3.深度NN：深度置信NN，卷积NN，循环NN，LSTM NN

特点：
    输入层--》隐层-》全连接--》输出层
组成：
    结构：NN中变量可以是神经元连接的权重
    激励函数：大部分NN具有一个短时间尺度的动力学规则，
            来定义神经元如何根据其他神经元来改变自己的激励值
    学习规则：NN中权重如何随着时间而调整（反向传播算法）
=============================================================
算法：神经网络
策略：交叉熵损失  信息熵
优化：反向传播，其实就是梯度下降
"""