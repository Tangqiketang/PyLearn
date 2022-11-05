import tensorflow as tf
import os

"用哪个图创建的，之后就属于哪个图"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"g和默认的tf.graph是两个图对象"
graph = tf.get_default_graph()
print("默认的图",graph)
a = tf.constant(5.0)

"创建一个新的图"
g = tf.Graph()
print("新的图:",g)

with g.as_default():
    b = tf.constant(11.0)
    print("新的图:",b.graph)

c = tf.constant(5.0)
d = tf.constant(6.0)
sum1 = tf.add(c,d)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("默认图a",a.graph)
    print("新的图b",b.graph)
    print("默认图c", c.graph)
    print("session", sess.graph)
    print(sess.run([c,d,sum1]))

"创建入参占位符,创建一个不固定的矩阵参数" \
"   从别人那里读取的数据是不固定的 "\
"feed_dict允许调用者覆盖图中指定张量的值"  \
""
plt = tf.placeholder(tf.float32,shape=[None,3])
print(plt)
with tf.Session() as sess:
    print(sess.run(plt,feed_dict={plt:[[1,2,3],[4,5,6]]}))