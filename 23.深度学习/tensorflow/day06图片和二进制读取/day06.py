import tensorflow as tf
import os as os

######图片读取
"""https://www.cs.toronto.edu/~kriz/cifar.html"""


def pic_read(filelist):
    """读取燕子图片转换成张量
    返回每张图片的张量"""
    ##1.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    ##2.构造阅读器去读取图片内容，默认读取一张
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)
    ##3.对数据解码
    image = tf.image.decode_jpeg(value)
    print("解码之后图片:",image)
    ##4.处理图片的大小，统一大小
    image_resize = tf.image.resize_images(image,[200,200])
    print("统一像素之后图片大小:",image_resize)
    ##批处理 注意：在批处理的时候,一定要把样本的形状固定
    image_resize.set_shape([200,200,3])
    image_batch = tf.train.batch([image_resize],batch_size=20,num_threads=1,capacity=20)
    return image_batch


if __name__ == "__main__":
    ###读取文件列表
    filename = os.listdir("./picdata/")
    print("文件名:",filename)
    filelist = [os.path.join("./picdata/",file) for file in filename if file[-3:]=="jpg"]
    ####读取文件op
    image_batch = pic_read(filelist)

    with tf.Session() as sess:
        # 开启读文件的线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)

        ##运行op， 打印读取的内容
        print(sess.run([image_batch]))

        ##回收子线程
        coord.request_stop()
        coord.join(threads)