
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

结合图片：
    1.样品特征×权重+bias，经过softmax后每个样本对应每个输出都有一个概率值。tf.matmul(a,b,)+bias
    2.把概率值和真实结果通过交叉熵公司计算，得到每个样品的交叉熵损失。损失值列表=tf.nn.softmax_cross_entropy_withlogits(labels=真实值,logits=样本加权之后的值)
    3.根据损失值列表获取平均值   tf.reduce_mean(损失值列表)
    ===》开始优化
    4.梯度下降  梯度下降op=tf.train.GradientDescentOptimizer(learning_rate)
    5.精确值计算 tf.equal()
"""

from tensorflow.examples.tutorials.mnist import input_data
###从本地文件读取
mnist = input_data.read_data_sets("./mnist/input_data/",one_hot=True)

print("第0个样本的特征矩阵：",mnist.train.images[0])
print("第0个样本的标签：",mnist.train.labels[0])
##获取50个样品
##mnist.train.next_batch(50)