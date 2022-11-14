"""
tensorflow分布式：单机单卡或多机多卡

参数服务器parameter server: ps
    协调存储，更新参数
工作服务器worker: 默认每个worker组中第一台作为master

1.对集群当中的参数服务器ps，工作服务器worker进行配置
    cluster = tf.train.ClusterSpec({"ps":[ip:port],"worker":[ip:port]})
2.使用配置创建对应的服务
    server = tf.train.Server(cluster,job_name,task_index)
        ps服务器join()，什么都不用干只需要等待work传递参数
        work服务器运行模型，初始化会话。指定一个默认的work老大去做
3.work如何使用设备：
    直接使用方式（非分布式）：
    with tf.device("/job:worker/task:0/gpu:0")
    分布式使用方式：
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:0/gpu:0",
        cluster=cluster
    ))

命令：
    workon tensorflow指定某一台机子以worker运行tensorflow

"""

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name","worker","启动服务的类型 ps or worker")
tf.app.flags.DEFINE_string("task_index",0,"指定ps或worker中哪一台服务器以task:0,task:1")


def main(args):
    ###1.定义全局计数的op,对应了后面的hooks=[tf.train.StopAtStepHook(last_step=200)]
    global_step = tf.contrib.framework.get_or_create_global_step()

    ###对集群当中的参数服务器ps，工作服务器worker进行配置.
    cluster = tf.train.ClusterSpec({"ps":["192.168.40.1:2223"],"worker":["192.168.40.134:2222"]})
    ##创建不同的服务，ps或work
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        ###参数服务器只负责更新保存参数，只需要等待worker传递参数
        server.join()
    else:
        ###指定设备运行,在分布式环境下运行
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0/cpu:0"),
                       cluster=cluster):
            x = tf.Variable([[1,2,3,4]])
            w = tf.Variable([[2],[2],[2],[2]])
            mat = tf.matmul(x,w)
        ##创建分布式的会话。之前的session不行
        with tf.train.MonitoredTrainingSession(
                master="grpc://192.168.40.134:2222",  ##指定master
                is_chief=(FLAGS.task_index ==0),  ####是否为master，这里指定第一台0为master
                config=tf.ConfigProto(log_device_placement=True),
                hooks=[tf.train.StopAtStepHook(last_step=200)]  ###训练几次
        ) as mon_sess:
            while not mon_sess.should_stop(): ##是否异常停止
                print(mon_sess.run(mat))


if __name__ == '__main__':
    tf.app.run()
