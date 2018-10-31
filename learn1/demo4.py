#-*-coding:utf-8-*- 
__author__ = 'Administrator' 

import tensorflow as tf

# TensorBoard
# 为了更方便 TensorFlow 的建模和调优，Google 还为 TensorFlow 开发了一款可视化的工具：TensorBoard，将我们第一个Demo的代码稍微改造一下，就可以使用 TensorBoard更加直观的理解 TensorFlow 的训练过程。


# 创建节点时设置name，方便在图中识别
W = tf.Variable([0], dtype=tf.float32, name='W')
b = tf.Variable([0], dtype=tf.float32, name='b')

# 创建节点时设置name，方便在图中识别
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

# 线性模型
linear_model = W * x + b

# 损失模型隐藏到loss-model模块
with tf.name_scope("loss-model"):
    loss = tf.reduce_sum(tf.square(linear_model - y))
    # 给损失模型的输出添加scalar，用来观察loss的收敛曲线
    tf.summary.scalar("loss", loss)

optmizer = tf.train.GradientDescentOptimizer(0.001)

train = optmizer.minimize(loss)

x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 调用 merge_all() 收集所有的操作数据
merged = tf.summary.merge_all()

# 模型运行产生的所有数据保存到 /tmp/tensorflow 文件夹供 TensorBoard 使用
# writer = tf.summary.FileWriter('/tmp/tensorflow', sess.graph)
writer = tf.summary.FileWriter('D:\python', sess.graph)


# 训练10000次
for i in range(10000):
    # 训练时传入merge
    summary, _ = sess.run([merged, train], {x: x_train, y: y_train})
    # 收集每次训练产生的数据
    writer.add_summary(summary, i)

curr_W, curr_b, curr_loss = sess.run(
    [W, b, loss], {x: x_train, y: y_train})

print("After train W: %s b %s loss: %s" % (curr_W, curr_b, curr_loss))
#
# 运行完上面的代码后，训练过程产生的数据就保存在 /tmp/tensorflow 文件夹了，我们可以在命令行终端运行下面的命令启动 TensorBoard：
# # 通过 --logdir 参数设置我们存放训练数据的目录
# $ tensorboard --logdir /tmp/tensorflow

# 然后在浏览器中打开 http://localhost:6006 页面就可以看到我们的模型数据了。
# 首先在 SCALARS 页面我们可以看到我们通过 tf.summary.scalar("loss", loss)设置的loss收敛曲线，从曲线图中可以看出在训练了大概2000次的时候loss就已经收敛的差不多了。
#
# http://127.0.0.1:6006/
