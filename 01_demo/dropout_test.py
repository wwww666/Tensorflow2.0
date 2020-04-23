# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 13:28
# @Author  : wwwzk
# @FileName: dropout_test.py
'''
从零实现dropout层，防止过拟合
'''
import tensorflow as tf
from tensorflow import nn,losses
import tensorflow,keras as ks
from tensorflow.keras.layers import Dropout,Flatten,Dense

# 自定义dropout函数
def dropout(X,drop_prob):
    assert 0<=drop_prob<=1
    keep_prob=1-drop_prob
    if keep_prob==0:
        return tf.zeros_like(X)
    mask=tf.random.uniform(shape=X.shape,minval=0,maxval=1)<keep_prob
    return tf.cast(mask,dtype=tf.float32)*tf.cast(X,dtype=tf.float32)/keep_prob
X=tf.reshape(tf.range(0,16),shape=(2,8))
print(X)
# print(dropout(X,0))
print(dropout(X,0.5))
# print(dropout(X,1))


