# coding:utf-8

import tensorflow as tf

##定义一个简单的计算图，实现两个常量相加的操作
with tf.name_scope('graph') as scope:
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    add = tf.add(a, b, name='add')
sess = tf.Session()

##生成一个写日志的writer，并将当前的tensorflow计算图写入日志
writer = tf.summary.FileWriter('Dr:/logs', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)