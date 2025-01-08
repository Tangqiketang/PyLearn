"""
1.卷积
   conv_1 = tf.nn.conv2d(输入特征[batch,height,width,channel]，
                 观察窗口filter权重[height,width,输入通道黑白1彩色3,输出通道为同时几个一个观察] ，
                 零填充padding="same"表示输出图片大小不变
                 步长strides=[1,stride,stride,1]表示上下左右都统一1)
    结果形状为[None样本数量,height,weight,filter的个数]
2.激活relu
    relu_1 = tf.nn.relu(conv_1)
3.池化pool:取出某一块区域最大值，比如4个变1个
    pool_1 = tf.nn.max_pool(relu_1,
                            ksize[,,,]，
                            零填充padding="same"表示输出图片大小不变
                            步长strides[1,stride,stride,1])
4.全链接层



激活函数：增加网络的非线性分割能力
池化：最大池化，平均池化，通过取局部区域的最大值或平均值来减少维度，如5*5-》3*3，减少小的尺寸差异。

神经网络中，用RELU替代sigmod 。1.计算量2.梯度爆炸

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variables(shape):
    """返回随机初始化权重"""
    return tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
def bias_varables(shape):
    """返回初始化值为0的偏置"""
    return tf.Variable(tf.constant(0.0,shape=shape))


def model():
    """自定义卷积模型 原始图片28×28=784个特征的大小"""
    ##1.准备数据占位符
    x = tf.placeholder(tf.float32,[None,784])
    y_true = tf.placeholder(tf.int32,[None,10])
    """2.定义第一层卷积：1.观察窗口Filter5×5同时有32个一起观察
                    2.步长strides为1 
                    3.零填充padding="same"表示 """
    ##卷积层输出按照公式计算本来应该是（28+2×填充-观察窗口filter的长度）/步长+1 = （28+2×padding-5）/1+1,
    # 但是tensorflow中padding=same时，卷积层输出和输入大小一样，所以还是28×28

    ## 观察窗口filter就是权重[height,width,输入通道黑白1如果N个叠加就为N,输出通道为同时几个观察] ，
    w_conv1 = weight_variables([5,5,1,32])
    ## 对应32个偏置项
    b_conv1 = bias_varables([32])
    ##进行卷积计算。输入x原先为一维的形状，进行 符合卷积计算的改变[batch,height,width,channel]
    x_reshape = tf.reshape(x,[-1,28,28,1])##改变形状不知道不是None而是-1
    x_2d1 = tf.nn.conv2d(x_reshape,w_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1   ##[None,28,28,32]
    print("x_2d1",x_2d1)
    ##激活
    x_relu1 = tf.nn.relu(x_2d1)
    ###池化 2*2 , [None, 28, 28, 32] 卷积得到的32个观测结果 --变成-->[None, 14, 14, 32].
    x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    """第二次卷积：1.观察窗口Filter5×5同时有64个一起观察
                    2.步长strides为1 
                    3.零填充padding="same"表示 """
   ##第一次的结果为32个重叠的矩阵。
    w_conv2 =weight_variables([5,5,32,64])
    b_conv2 = bias_varables([64])
    x_2d2 = tf.nn.conv2d(x_pool1,w_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2
    print("x_2d2", x_2d2)
    x_relu2 = tf.nn.relu(x_2d2)
    ## [None,14,14,64]  -->[None,7,7,64]
    x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    ##全连接层
    x_fc_reshape = tf.reshape(x_pool2,[-1,7*7*64])
    weight_fc = weight_variables([7*7*64,10])
    bias_fc = bias_varables([10])
    y_predict = tf.matmul(x_fc_reshape,weight_fc)+bias_fc

    return x,y_true,y_predict


def conv_fc():
    """卷积全连接层"""
    mnist = input_data.read_data_sets("./mnist/input_data/",one_hot=True)
    x,y_true,y_predict = model()
    ###平均交叉熵损失
    sum_reduce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(sum_reduce)
    ###梯度下降
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    ###计算准确率
    equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            mnist_x,mnist_y = mnist.train.next_batch(50)
            sess.run(train_op,feed_dict={x:mnist_x,y_true:mnist_y})
            print("训练第%d步,准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
    return None

if __name__=="__main__":
    conv_fc()