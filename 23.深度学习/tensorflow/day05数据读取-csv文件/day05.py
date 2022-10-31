import tensorflow as tf

#模拟同步先处理数据，然后才取数据训练

#1.首先定义队列/放入数据
Q = tf.FIFOQueue(3,tf.float32)
enq_many = Q.enqueue_many([[0.1,0.2,0.3],])

#2.定义一些读取数据，，取数据，+1,入队列。  注意有依赖关系的op属于同一个，只需要执行en_q就行
out_q = Q.dequeue()
data = out_q + 1
en_q = Q.enqueue(data)

with tf.Session() as sess:
    #初始化队列，放入一个初始化张量
    sess.run(enq_many)
    #处理数据
    for i in range(100):
        sess.run(en_q)
    ##训练数据
    for i in range(Q.size().eval()):
        print(sess.run(Q.dequeue()))

