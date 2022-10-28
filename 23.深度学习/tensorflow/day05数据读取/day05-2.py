import tensorflow as tf
import os as os

######文件读取

def csv_read(filelist):
    """
    读取csv文件
    :param filelist:  文件路径+名字列表
    :return:
    """
    ##1.构建要去读的文件队列
    file_queue = tf.train.string_input_producer(filelist)
    #2.构造csv阅读器读取队列数据,一行一行
    reader = tf.TextLineReader()

    key,value = reader.read(file_queue)

    #3.对每行内容value解码. 给默认值
    records = [["None"],["None"]]
    example,label = tf.decode_csv(value,record_defaults=records)

    ##4.想要读取多个数据 管道批处理。 每次取9个
    example_batch,label_batch = tf.train.batch([example,label],batch_size=9)

    return example_batch,label_batch



if __name__ == "__main__":
    ###读取文件列表
    filename = os.listdir("./csvdata/")
    print(filename)
    filelist = [os.path.join("./csvdata/",file) for file in filename if file[-3:]=="csv"]
    ####读取文件op
    example,lable = csv_read(filelist)

    with tf.Session() as sess:
        # 开启读文件的线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)

        ##运行op， 打印读取的内容
        print(sess.run([example,lable]))

        ##回收子线程
        coord.request_stop()
        coord.join(threads)