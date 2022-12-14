import tensorflow as tf
import os


"""先把图片按照cifarRead读出label image，写入tfrecord文件，再读取文件"""

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir","./cifar10/cifar-10-batches-py/","二进制文件目录")
tf.app.flags.DEFINE_string("cifar_tfrecords","./cifar10/tmp/cifar.tfrecords","存tfrecords的文件")

class TfRecord(object):
    def __init__(self,filelist):
        self.file_list = filelist
        ##定义读取图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        #二进制文件每张图片的字节数=一个分类标记+像素×通道
        self.label_bytes =1 ##第一位是分类标志
        self.image_bytes = self.height * self.width *self.channel
        self.bytes = self.image_bytes+self.label_bytes

    def read_from_tfrecords(self):
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])
        # 2、构造文件阅读器，读取内容example,value=一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        # 3、解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
        # 4、解码内容, 如果读取的内容格式是string需要解码， 如果是int64,float32不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)
        # 固定图片的形状，方便与批处理
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label = tf.cast(features["label"], tf.int32)
        print(image_reshape, label)
        # 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch


    def write_ro_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和目标值存进tfrecords
        :param image_batch: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        # 1、建立TFRecord存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 2、循环将所有样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i个图片数据的特征值和目标值
            image = image_batch[i].eval().tostring()
            label = int(label_batch[i].eval()[0])
            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            # 写入单独的样本
            writer.write(example.SerializeToString())
        # 关闭
        writer.close()
        return None

if __name__ == "__main__":
    file_names = os.listdir(FLAGS.cifar_dir)
    filelist = [os.path.join(FLAGS.cifar_dir,file) for file in file_names if file[-3:]=="bin"]
    ###创建自己的类，并调用
    cf = TfRecord(filelist)

    ##image_batch,label_batch = cf.read_and_decode()
    image_batch, label_batch = cf.read_from_tfrecords()

    with tf.Session() as sess:
        #定义一个线程协调器
        coord = tf.train.Coordinator()
        ##读文件线程
        threads = tf.train.start_queue_runners(sess,coord=coord)

        ##写入tfrecord tensorflow自带的字典格式数据结构
        ##cf.write_ro_tfrecords(image_batch, label_batch)

        print(sess.run([image_batch,label_batch]))

        ##回收子线程
        coord.request_stop()
        coord.join(threads)


