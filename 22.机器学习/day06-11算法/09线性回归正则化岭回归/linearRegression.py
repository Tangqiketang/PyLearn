"""
w1x2+w2x2+w3x3+....w_nx_n+ bias

[[1,2,3,4,5]  [[20]
[5,2,3,1,1]]  [30]]
====================================================================
算法：09线性回归正则化岭回归
策略：均方误差。
优化：梯度下降

1. 准备好 特征 和目标值
2. 建立模型 随机初始化准备一个权重w,一个偏置项b
    y_predict = x w + b
3. 求损失函数，误差。均方差是和真实值的差。  假设100个
loss 均方误差 (y1-y1')^2+......(y_100-y_100')^2 /100

4.梯度下降优化损失过程 指定学习率
    m 行：m个样本
    n 列：每个样本n个特征值

    矩阵相乘  （m n）(n 1)  ==> (m,1)+bias
"""

import tensorflow as tf

#####一维特征
def myregression():
    """
    这里是自己根据函数模拟的数据。真实数据的话应该是要做 标准化处理的
    :return:
    """

    #创建假数据  y=0.7x+0.8
    x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name="x_data")
    y_true = tf.matmul(x,[[0.7]])+0.8

    ##随机创建一个权重矩阵 和一偏置项 和预测结果函数
    weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name="w")
    bias = tf.Variable(0.0,name="b")
    y_predict = tf.matmul(x,weight) + bias

    ##损失函数 为均方差. 对损失函数梯度下降.学习率一般0~10,这里选0.1
    loss = tf.reduce_mean(tf.square(y_true-y_predict))
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    ##定义初始化变量op
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        print("随机初始化参数权重为:%f，偏置为：%f" % (weight.eval(),bias.eval()))
        for i in range(100):
            sess.run(train_op)
            print("优化%f次后，loss为%s，参数权重为:%f，偏置为：%f" % (i,loss,weight.eval(), bias.eval()))
    return None

if __name__ == "__main__":
    myregression()
