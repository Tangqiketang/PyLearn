import tensorflow as tf

######多线程读取队列

##创建队列
Q = tf.FIFOQueue(1000,tf.float32)
##定义要做的操作  循环值，+1,放入队列
var = tf.Variable(0.0)
data = tf.assign_add(var,tf.constant(1.0))
en_q = Q.enqueue(data)

##3.定义队列管理器op,指定多少个子线程从哪个队列拿，子线程要干哪些操作op
qr = tf.train.QueueRunner(Q,enqueue_ops=[en_q]*2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    ##开启线程管理器
    coord = tf.train.Coordinator()
    ##真正开启子线程
    threads = qr.create_threads(sess,coord=coord,start=True)

    #主线程不断读取数据训练
    for i in range(300):
        print(sess.run(Q.dequeue()))
    ##回收线程
    coord.request_stop()
    coord.join(threads)