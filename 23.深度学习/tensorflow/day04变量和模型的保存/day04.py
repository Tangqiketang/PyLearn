"""可视化学习 Tensorboard"""
import tensorflow as tf


a=tf.constant(3.0,name="a")
b=tf.constant(4.0,name="b")

"""
1.变量op可以持久化保存，普通张量op不行
2.当定义一个变量op的时候，一定要在会话中去运行初始化
3.name参数 在tensorboard使用的时候显示名字"""
c=tf.add(a,b,name="my_add_op")


#变量.变量必须初始化
a2= tf.constant([1,2,3,4,5])
var2 = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0),name="my_random_variable")


print(a2,var2)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    #把程序的图结构写入事件文件. 页面需要刷新才会更新
    fileWriter = tf.summary.FileWriter("./tmp/summary/test/",graph=sess.graph)
    # 控制台开启可视化。切换到python环境运行 tensorboard --logdir="./tmp/summary/test/",访问提示的url

    print(sess.run([a2,var2,c]))