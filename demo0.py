#-*-coding:utf-8-*- 
__author__ = 'Administrator' 

# python3环境下输入
import tensorflow as tf
h = tf.constant('Hello, Tensorflow!')
s = tf.Session()
print(s.run(h))
