
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
    5.精确值计算.获取每个样品精确值列表，预测相同的为1,得到列表计算平均值
        equallist = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equallist,tf.float32))
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def full_connected():
    ##从本地文件中获取真实的手写字数据
    mnist = input_data.read_data_sets("./mnist/input_data/",one_hot=True)
    ##1. 建立数据特征值 x，真实目标值，占位符
    with tf.variable_scope("data"):
        x =  tf.placeholder(tf.float32,shape=[None,784])
        y_true = tf.placeholder(tf.float32,[None,10])

    ##2.建立权重和偏置项
    with tf.variable_scope("fc_model"):
        weight = tf.Variable(tf.random_normal([784,10],stddev=1.0,mean=0.0),name="w")
        bias = tf.Variable(tf.constant(0.0,shape=[10]))
        y_predict = tf.matmul(x,weight)+bias

    ##3.求出损失. softmax_交叉——熵--全链接
    with tf.variable_scope("soft_cross"):
        sum_reduce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict)
        loss = tf.reduce_mean(sum_reduce)

    ##4.梯度下降优化op
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    with tf.variable_scope("acc"):
        ##5.计算准确率. 获取真实值中[0.1,0.2,0.5,0.1,xxx]中概率最大的那个，和预测值中最大的那个。比较是否相等
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    #####收集变量
    tf.summary.scalar("losses",loss)
    tf.summary.scalar("acc",accuracy)
    tf.summary.histogram("weightes",weight)
    tf.summary.histogram("biases",bias)
    merged = tf.summary.merge_all()

    ###创建保存数据
    saver = tf.train.Saver()
    ##定义初始化变量的op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        ###建立文件写入.tensorboard --logdir="./mnist/tmp/summary/test",访问提示的url
        filewriter = tf.summary.FileWriter("./mnist/tmp/summary/test/",graph=sess.graph)

        for i in range(20000):
            ## 获取真实数据中特征值和目标值
            mnist_x,mnist_y = mnist.train.next_batch(5000)
            ##运行训练op
            sess.run(train_op,feed_dict={x:mnist_x,y_true:mnist_y})
            print("x训练第%d步，准确率为:%f" % (i,sess.run(accuracy,feed_dict={x:mnist_x,y_true:mnist_y})))
            ##写入每步的训练到文件
            summary = sess.run(merged,feed_dict={x:mnist_x,y_true:mnist_y})
            filewriter.add_summary(summary,i)

        saver.save(sess,"./mnist/tmp/ckpt/fc_model")

    return None


if __name__ == "__main__":
    full_connected()