"""
目标值[A,B,C,D]

"""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("captcha_dir", "./tfrecords/captcha.tfrecords", "验证码数据的路径")
tf.app.flags.DEFINE_integer("batch_size", 100, "每批次训练的样本数")
tf.app.flags.DEFINE_integer("label_num", 4, "每个样本的目标值数量")
tf.app.flags.DEFINE_integer("letter_num", 26, "每个目标值取的字母的可能心个数")


def read_and_decode():
    """从tfrecord格式的文件中获取图片和标签并解码"""
    ##1.从文件队列中读取图片和标签
    file_queue = tf.train.input_producer([FLAGS.captcha_dir])
    reader = tf.TFRecordReader()
    key,value = reader.read(file_queue)
    ##2.解析某一行,single一个样本。
    features = tf.parse_single_example(value,
            features={"image":tf.FixedLenFeature([],tf.string),"label":tf.FixedLenFeature([],tf.string)})
    # 4、解码内容, 如果读取的内容格式是string需要解码， 如果是int64,float32不需要解码
    image = tf.decode_raw(features["image"], tf.uint8)
    label = tf.decode_raw(features["label"], tf.uint8)
    print("image.before.reshape",image)
    print("label.before.reshape",label)
    # 固定图片的形状，方便与批处理
    image_reshape = tf.reshape(image, [20,80,3])
    label_reshape = tf.reshape(label,[4]) ## [13,25,15,15]目标值

    print("image.after.reshape",image_reshape)
    print("label.after.reshape",label_reshape)
    ###5.按照单个样本解析的方式，实行批量操作返回
    image_batch,label_batch = tf.train.batch([image_reshape,label_reshape],batch_size=FLAGS.batch_size,num_threads=1,capacity=FLAGS.batch_size)
    return image_batch,label_batch

def weight_variable(shape):
    """初始化权重函数"""
    return tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))

def bias_variable(shape):
    """初始化偏重函数"""
    return tf.Variable(tf.constant(0.0,shape=shape))

def fc_model(image_batch):
    """获取批量图片的预测值"""
    ##将图片转行成2维
    image_reshape = tf.reshape(image_batch,[-1,20*80*3]) ####-1表示不知道有多少个样本，这里其实可以用100来替换
    weights = weight_variable([20*80*3,4*26]) ##四个位置，每个位置26种可能
    bias = bias_variable([4*26])
    y_predict = tf.matmul(tf.cast(image_reshape,tf.float32),weights)+bias
    return y_predict ##[n,4*26]

def label_to_onehot(label_batch):
    """
        把[100,4]的目标值变成 [100,4,26]的批量onehot.depth表示一共多少种分类
    """
    print("lable.before.onehot:",label_batch)
    onehot_label = tf.one_hot(label_batch,depth=FLAGS.letter_num,on_value=1.0,axis=2)
    print("lable.after.onehot:",onehot_label)
    return onehot_label

def captcha_fc():
    """一层全链接"""
    image_batch,label_batch = read_and_decode()
    y_predict = fc_model(image_batch)    ##[-1,4*26]
    y_true = label_to_onehot(label_batch) ##[-1,4,26]


    ###用softmax计算交叉熵
    sum_reduce = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(y_true,[-1,4*26]),
        logits=y_predict
        )
    loss = tf.reduce_mean(sum_reduce)
    ##梯度下降
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    ###求出样品每批次预测的准确率是多少 [1,4,26]中第二个位置26处
    equal_list = tf.equal(tf.argmax(y_true,axis=2),tf.argmax(tf.reshape(y_predict,[-1,4,26]),axis=2))
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    ##
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        ##定义线程协调器和开启线程
        coord = tf.train.Coordinator()
        #开启线程去运行读取文件操作
        threads = tf.train.start_queue_runners(sess,coord=coord)

        for i in range(2000):
            sess.run(train_op)
            print("第%d批准备率为:%f" % (i,accuracy.eval()))

        coord.request_stop()
        coord.join(threads)
    return None









if __name__ == "__main__":
    captcha_fc()

