"""
w1x2+w2x2+w3x3+....w_nx_n+ bias

[[1,2,3,4,5]  [[20]
[5,2,3,1,1]]  [30]]
====================================================================
算法：线性回归
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
import os as os

###定义全局参数，放入全局变量FLAGS
tf.app.flags.DEFINE_string("model_dir","./tmp/checkpoint/model","模型保存的目录,model是文件")
tf.app.flags.DEFINE_integer("max_step",100,"模型训练的次数")

FLAGS = tf.app.flags.FLAGS


###二维特征+变量域+文件保存 收集变量
def myregression():
    ##创建作用域，用于tensorboard更清晰
    with tf.variable_scope("data"):
        #创建假数据  y=0.7x+0.1z+0.8
        x = tf.random_normal([100,2],mean=1.75,stddev=0.5,name="x_data")
        y_true = tf.matmul(x,[[0.7],[0.1]])+0.8

    with tf.variable_scope("model"):
        ##随机创建一个权重矩阵 和一偏置项 和预测结果函数
        weight = tf.Variable(tf.random_normal([2,1],mean=0.0,stddev=1.0),name="w")
        bias = tf.Variable(0.0,name="b")
        y_predict = tf.matmul(x,weight) + bias

    with tf.variable_scope("loss"):
        ##损失函数 为均方差. 对损失函数梯度下降.学习率一般0~10,这里选0.1
        loss = tf.reduce_mean(tf.square(y_true-y_predict))
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    ###收集tensor用于显示
    tf.summary.scalar("losses",loss) ##单值
    tf.summary.histogram("weights",weight) ##高维度变量
    ###tf.summary.image(name="",tensor) ##收集图像
    ##合并变量tensor的op
    merged = tf.summary.merge_all()

    #定义一个保存模型的实例
    saver = tf.train.Saver()

    ##定义初始化变量op
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print("随机初始化参数权重为:%s，偏置为：%f" % (weight.eval(),bias.eval()))

        ##建立事件文件
        fileWriter = tf.summary.FileWriter("./tmp/summary/test",graph=sess.graph)

        ##i加载模型，覆盖模型中随机定义的参数，从上次训练的参数开始
        if os.path.exists("./tmp/checkpoint"):
            saver.restore(sess, FLAGS.model_dir)

        for i in range(FLAGS.max_step):
            #训练
            sess.run(train_op)

            ##运行合并的tensor 用于tensorboard显示
            summary = sess.run(merged)
            fileWriter.add_summary(summary,i)

            print("优化%f次后，loss为%s，参数权重为:%s，偏置为：%f" % (i,loss,weight.eval(), bias.eval()))

        saver.save(sess,FLAGS.model_dir)
    return None

if __name__ == "__main__":
    myregression()
