#-*-coding:utf-8-*- 
__author__ = 'Administrator' 

#
# TensorFlow 高级训练模型
# tf.estimator是TensorFlow提供的高级库，提供了很多常用的训练模型，可以简化机器学习中的很多训练过程，如：
#
# 运行训练循环
# 运行评估循环
# 管理训练数据集
# 评估模型
# 前面的demo中我们构建了一个线性模型，通过使用一组实验数据训练我们的线性模型，我们得到了一个自认为损失最小的最优模型，根据训练结果我们的最优模型可以表示为下面的方程：
# y=2.98x+2.07
# y=2.98x+2.07
#
# 但是这个我们自认为的最优模型是否会一直是最优的？我们需要通过一些新的实验数据来评估(evaluation)模型的泛化性能(generalization performance)，如果新的实验数据应用到到这个模型中损失值越小，那么这个模型的泛化性能就越好，反之就越差。下面的demo中我们也会看到怎么评估模型。
# 使用LinearRegressor
# 前面我们构建了一个线性模型，通过训练得到一个线性回归方程。tf.estimator中也提供了线性回归的训练模型tf.estimator.LinearRegressor，下面的代码就是使用LinearRegressor训练并评估模型的方法：


# 我们会用到NumPy来处理各种训练数据
import numpy as np
import tensorflow as tf

# 创建一个特征向量列表，该特征列表里只有一个特征向量，
# 该特征向量为实数向量，只有一个元素的数组，且该元素名称为 x，
# 我们还可以创建其他更加复杂的特征列表
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 创建一个LinearRegressor训练器，并传入特征向量列表
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 保存训练用的数据
x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

# 保存评估用的数据
x_eavl = np.array([2., 5., 7., 9.])
y_eavl = np.array([7.6, 17.2, 23.6, 28.8])

# 用训练数据创建一个输入模型，用来进行后面的模型训练
# 第一个参数用来作为线性回归模型的输入数据
# 第二个参数用来作为线性回归模型损失模型的输入
# 第三个参数batch_size表示每批训练数据的个数
# 第四个参数num_epochs为epoch的次数，将训练集的所有数据都训练一遍为1次epoch
# 低五个参数shuffle为取训练数据是顺序取还是随机取
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)

# 再用训练数据创建一个输入模型，用来进行后面的模型评估
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)

# 用评估数据创建一个输入模型，用来进行后面的模型评估
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eavl}, y_eavl, batch_size=2, num_epochs=1000, shuffle=False)

# 使用训练数据训练1000次
estimator.train(input_fn=train_input_fn, steps=1000)

# 使用原来训练数据评估一下模型，目的是查看训练的结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)

# 使用评估数据评估一下模型，目的是验证模型的泛化性能
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %s" % eval_metrics)
# 评估数据的loss比训练数据还要小，说明我们的模型泛化性能很好
