#-*-coding:utf-8-*-
__author__ = 'Administrator'

import input_data
import tensorflow as tf

#读取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()


#构建cnn网络结构
#自定义卷积函数（后面卷积时就不用写太多）
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#自定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#设置占位符，尺寸为样本输入和输出的尺寸
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1])

#设置第一个卷积层和池化层
w_conv1=tf.Variable(tf.truncated_normal([3,3,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#设置第二个卷积层和池化层
w_conv2=tf.Variable(tf.truncated_normal([3,3,32,50],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[50]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#设置第一个全连接层
w_fc1=tf.Variable(tf.truncated_normal([7*7*50,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*50])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout（随机权重失活）
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#设置第二个全连接层
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#建立loss function，为交叉熵
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))
#配置Adam优化器，学习速率为1e-4
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

#建立正确率计算表达式
correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#开始喂数据，训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print("step %d,train_accuracy= %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#训练之后，使用测试集进行测试，输出最终结果
print("test_accuracy= %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))

#
# C:\Python36_64\python.exe D:/python/python_tensorflow/learn2/demo1.py
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# 2018-10-31 16:12:37.024269: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# step 0,train_accuracy= 0.12
# step 100,train_accuracy= 0.74
# step 200,train_accuracy= 0.9
# step 300,train_accuracy= 0.82
# step 400,train_accuracy= 0.94
# step 500,train_accuracy= 0.92
# step 600,train_accuracy= 0.98
# step 700,train_accuracy= 0.96
# step 800,train_accuracy= 0.92
# step 900,train_accuracy= 1
# step 1000,train_accuracy= 0.92
# step 1100,train_accuracy= 0.98
# step 1200,train_accuracy= 0.92
# step 1300,train_accuracy= 1
# step 1400,train_accuracy= 0.98
# step 1500,train_accuracy= 0.96
# step 1600,train_accuracy= 0.96
# step 1700,train_accuracy= 0.96
# step 1800,train_accuracy= 0.96
# step 1900,train_accuracy= 0.98
# step 2000,train_accuracy= 0.98
# step 2100,train_accuracy= 1
# step 2200,train_accuracy= 0.98
# step 2300,train_accuracy= 0.96
# step 2400,train_accuracy= 0.96
# step 2500,train_accuracy= 0.98
# step 2600,train_accuracy= 0.98
# step 2700,train_accuracy= 0.94
# step 2800,train_accuracy= 0.98
# step 2900,train_accuracy= 0.94
# step 3000,train_accuracy= 1
# step 3100,train_accuracy= 0.96
# step 3200,train_accuracy= 0.98
# step 3300,train_accuracy= 1
# step 3400,train_accuracy= 1
# step 3500,train_accuracy= 0.98
# step 3600,train_accuracy= 0.96
# step 3700,train_accuracy= 1
# step 3800,train_accuracy= 1
# step 3900,train_accuracy= 0.96
# step 4000,train_accuracy= 1
# step 4100,train_accuracy= 0.98
# step 4200,train_accuracy= 1
# step 4300,train_accuracy= 1
# step 4400,train_accuracy= 0.98
# step 4500,train_accuracy= 0.96
# step 4600,train_accuracy= 1
# step 4700,train_accuracy= 0.98
# step 4800,train_accuracy= 0.98
# step 4900,train_accuracy= 0.98
# step 5000,train_accuracy= 1
# step 5100,train_accuracy= 0.94
# step 5200,train_accuracy= 1
# step 5300,train_accuracy= 0.92
# step 5400,train_accuracy= 1
# step 5500,train_accuracy= 0.98
# step 5600,train_accuracy= 0.98
# step 5700,train_accuracy= 0.98
# step 5800,train_accuracy= 0.96
# step 5900,train_accuracy= 0.98
# step 6000,train_accuracy= 0.98
# step 6100,train_accuracy= 0.98
# step 6200,train_accuracy= 1
# step 6300,train_accuracy= 0.98
# step 6400,train_accuracy= 1
# step 6500,train_accuracy= 0.98
# step 6600,train_accuracy= 0.94
# step 6700,train_accuracy= 1
# step 6800,train_accuracy= 1
# step 6900,train_accuracy= 0.98
# step 7000,train_accuracy= 1
# step 7100,train_accuracy= 1
# step 7200,train_accuracy= 1
# step 7300,train_accuracy= 1
# step 7400,train_accuracy= 1
# step 7500,train_accuracy= 1
# step 7600,train_accuracy= 0.98
# step 7700,train_accuracy= 1
# step 7800,train_accuracy= 1
# step 7900,train_accuracy= 1
# step 8000,train_accuracy= 1
# step 8100,train_accuracy= 0.98
# step 8200,train_accuracy= 1
# step 8300,train_accuracy= 0.98
# step 8400,train_accuracy= 1
# step 8500,train_accuracy= 1
# step 8600,train_accuracy= 0.96
# step 8700,train_accuracy= 1
# step 8800,train_accuracy= 1
# step 8900,train_accuracy= 1
# step 9000,train_accuracy= 1
# step 9100,train_accuracy= 0.98
# step 9200,train_accuracy= 0.94
# step 9300,train_accuracy= 1
# step 9400,train_accuracy= 0.98
# step 9500,train_accuracy= 1
# step 9600,train_accuracy= 1
# step 9700,train_accuracy= 1
# step 9800,train_accuracy= 1
# step 9900,train_accuracy= 1
# step 10000,train_accuracy= 1
# step 10100,train_accuracy= 0.98
# step 10200,train_accuracy= 1
# step 10300,train_accuracy= 1
# step 10400,train_accuracy= 1
# step 10500,train_accuracy= 1
# step 10600,train_accuracy= 1
# step 10700,train_accuracy= 0.98
# step 10800,train_accuracy= 1
# step 10900,train_accuracy= 1
# step 11000,train_accuracy= 1
# step 11100,train_accuracy= 0.98
# step 11200,train_accuracy= 0.98
# step 11300,train_accuracy= 1
# step 11400,train_accuracy= 1
# step 11500,train_accuracy= 1
# step 11600,train_accuracy= 1
# step 11700,train_accuracy= 1
# step 11800,train_accuracy= 1
# step 11900,train_accuracy= 1
# step 12000,train_accuracy= 1
# step 12100,train_accuracy= 1
# step 12200,train_accuracy= 1
# step 12300,train_accuracy= 1
# step 12400,train_accuracy= 1
# step 12500,train_accuracy= 1
# step 12600,train_accuracy= 0.98
# step 12700,train_accuracy= 1
# step 12800,train_accuracy= 1
# step 12900,train_accuracy= 1
# step 13000,train_accuracy= 1
# step 13100,train_accuracy= 1
# step 13200,train_accuracy= 1
# step 13300,train_accuracy= 1
# step 13400,train_accuracy= 1
# step 13500,train_accuracy= 1
# step 13600,train_accuracy= 1
# step 13700,train_accuracy= 1
# step 13800,train_accuracy= 1
# step 13900,train_accuracy= 1
# step 14000,train_accuracy= 0.98
# step 14100,train_accuracy= 1
# step 14200,train_accuracy= 1
# step 14300,train_accuracy= 1
# step 14400,train_accuracy= 1
# step 14500,train_accuracy= 1
# step 14600,train_accuracy= 1
# step 14700,train_accuracy= 1
# step 14800,train_accuracy= 1
# step 14900,train_accuracy= 1
# step 15000,train_accuracy= 1
# step 15100,train_accuracy= 1
# step 15200,train_accuracy= 1
# step 15300,train_accuracy= 1
# step 15400,train_accuracy= 1
# step 15500,train_accuracy= 1
# step 15600,train_accuracy= 1
# step 15700,train_accuracy= 1
# step 15800,train_accuracy= 1
# step 15900,train_accuracy= 1
# step 16000,train_accuracy= 1
# step 16100,train_accuracy= 1
# step 16200,train_accuracy= 1
# step 16300,train_accuracy= 1
# step 16400,train_accuracy= 1
# step 16500,train_accuracy= 0.98
# step 16600,train_accuracy= 1
# step 16700,train_accuracy= 1
# step 16800,train_accuracy= 1
# step 16900,train_accuracy= 1
# step 17000,train_accuracy= 1
# step 17100,train_accuracy= 1
# step 17200,train_accuracy= 1
# step 17300,train_accuracy= 1
# step 17400,train_accuracy= 1
# step 17500,train_accuracy= 1
# step 17600,train_accuracy= 1
# step 17700,train_accuracy= 1
# step 17800,train_accuracy= 1
# step 17900,train_accuracy= 1
# step 18000,train_accuracy= 1
# step 18100,train_accuracy= 1
# step 18200,train_accuracy= 0.98
# step 18300,train_accuracy= 1
# step 18400,train_accuracy= 1
# step 18500,train_accuracy= 0.98
# step 18600,train_accuracy= 1
# step 18700,train_accuracy= 0.98
# step 18800,train_accuracy= 1
# step 18900,train_accuracy= 1
# step 19000,train_accuracy= 1
# step 19100,train_accuracy= 1
# step 19200,train_accuracy= 1
# step 19300,train_accuracy= 1
# step 19400,train_accuracy= 1
# step 19500,train_accuracy= 1
# step 19600,train_accuracy= 1
# step 19700,train_accuracy= 1
# step 19800,train_accuracy= 1
# step 19900,train_accuracy= 1
# 2018-10-31 16:37:02.101067: W tensorflow/core/framework/allocator.cc:113] Allocation of 1003520000 exceeds 10% of system memory.
# 2018-10-31 16:37:03.367139: W tensorflow/core/framework/allocator.cc:113] Allocation of 250880000 exceeds 10% of system memory.
# 2018-10-31 16:37:03.702159: W tensorflow/core/framework/allocator.cc:113] Allocation of 392000000 exceeds 10% of system memory.
# test_accuracy= 0.9925
#

