import tensorflow as tf
import os

"""图片读取 https://www.cs.toronto.edu/~kriz/cifar.html"""

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir","./cifar10/cifar-10-batches-py/","二进制文件目录")
tf.app.flags.DEFINE_string("cifar_tfrecords","./cifar10/tmp/cifar.tfrecords","存tfrecords的文件")

class CifarRead(object):
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

    def read_and_decode(self):
        ##1.构造文件队列.shuffle=True表示可以乱序
        file_queue = tf.train.string_input_producer(self.file_list,shuffle=True)
        ##2.构造固定长度二进制文件读取器，每次读取每个样本的字节
        reader = tf.FixedLengthRecordReader(self.bytes)
        key,value = reader.read(file_queue)
        ##3.解码内容，二进制转为
        label_image = tf.decode_raw(value,tf.uint8)
        print("解码之后label_image：", label_image)

        ##4.分割出图片和标签数据，切除特征值和目标值
        label = tf.cast(tf.slice(label_image,[0],[self.label_bytes]), tf.int32)
        image = tf.slice(label_image,[self.label_bytes],[self.image_bytes])
        ##5.对图片的特征进行形状改变 [3072]-->[32,32,3]
        image_reshape = tf.reshape(image,[self.height,self.width,self.channel])

        ##批处理
        image_batch, label_batch = tf.train.batch([image_reshape,label],batch_size=10,num_threads=1,capacity=10)
        print(image_batch,label_batch)
        return image_batch,label_batch

if __name__ == "__main__":
    file_names = os.listdir(FLAGS.cifar_dir)
    filelist = [os.path.join(FLAGS.cifar_dir,file) for file in file_names if file[-3:]=="bin"]
    ###创建自己的类，并调用
    cf = CifarRead(filelist)
    image_batch,label_batch = cf.read_and_decode()

    with tf.Session() as sess:
        #定义一个线程协调器
        coord = tf.train.Coordinator()
        ##读文件线程
        threads = tf.train.start_queue_runners(sess,coord=coord)
        print(sess.run([image_batch,label_batch]))

        ##回收子线程
        coord.request_stop()
        coord.join(threads)


