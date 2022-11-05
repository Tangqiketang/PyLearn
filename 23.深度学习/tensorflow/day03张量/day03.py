import tensorflow as tf


a = tf.constant(5.0)

plt = tf.placeholder(tf.float32,shape=[2,3,4])

##with tf.Session() as sess:
print(a.shape)
print("--------------------")
print(plt.shape)
print("--------------------")
print(a.name)
print("--------------------")
print(a.op)


print("-----------------")


"""静态形状和动态形状 """
plt = tf.placeholder(tf.float32,[None,2])
plt.set_shape([3,2])  #设置后不能再次修改了

###动态重新生成
plt_reshape = tf.reshape(plt,[2,3])

print("plt:",plt)
print("plt_reshape:",plt_reshape)
with tf.Session() as sess:
    zero = tf.zeros([3,4],tf.float32)
    print(zero)
    print(zero.eval())
    xx = tf.cast([[1,2,3],[4,5,6]],tf.float32)
    print(xx.eval())


#变量.变量必须初始化
a2= tf.constant([1,2,3,4,5])
var2 = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0))
print(a2,var2)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([a2,var2]))






