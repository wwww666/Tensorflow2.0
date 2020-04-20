# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 11:46
# @Author  : wwwzk
# @FileName: weight_decay_test.py
'''
L2范数正则化权重衰减
'''
import tensorflow as tf
from tensorflow.keras import layers,optimizers,models,initializers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks
from liner_test import linreg,squared_loss,sgd
from fit_test import semilogy

# 定义初始化数据集，权重，偏重
n_train,n_test,num_input=20,100,200
true_w,true_b=tf.ones((num_input,1))*0.01,0.05
features = tf.random.normal(shape=(n_train+n_test,num_input))
labels=ks.backend.dot(features,true_w)+true_b
labels+=tf.random.normal(mean=0.01,shape=labels.shape)
train_features,test_features=features[:n_train,:],features[n_train:,:]
train_labels,test_labels=labels[:n_train],labels[n_train:]

# 定义随机初始化模型参数
def init_params():
    w=tf.Variable(tf.random.normal(mean=1,shape=(num_input,1)))
    b=tf.Variable(tf.zeros(shape=(1,)))
    return [w,b]
# 定义L2范数
def l2_penalty(w):
    return tf.reduce_sum(w**2)/2

# 定义超参数
batch_size,num_epochs,lr=1,100,0.003
#定义网络结构
net,loss=linreg,squared_loss
optimizer=ks.optimizers.SGD()
train_iter = tf.data.Dataset.from_tensor_slices((train_features,train_labels)).batch(batch_size).shuffle(batch_size)

# 训练模型
def fit_and_plot(lambd):
    w,b=init_params()
    train_ls,test_ls=[],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            with tf.GradientTape() as tape:
                l=loss(net(X,w,b),y)+lambd*l2_penalty(w)

            grads=tape.gradient(l,[w,b])
            sgd([w,b],lr,batch_size,grads)
        train_ls.append(tf.reduce_mean(loss(net(train_features,w,b),
                                            train_labels)).numpy())
        test_ls.append(tf.reduce_mean(loss(net(test_features,w,b),
                                           test_labels)).numpy())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', tf.norm(w).numpy())

fit_and_plot(lambd=0)
fit_and_plot(lambd=3)


