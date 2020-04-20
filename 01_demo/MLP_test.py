'''
自定义实现双层网络，实现Relu激活函数
'''
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from softmax_test  import train_che3
from tensorflow.keras.datasets.fashion_mnist import load_data

# 加载数据集；转换类型并做归一化处理
(x_train,y_train),(x_test,y_test)=load_data()
batch_size=256
x_train=tf.cast(x_train,tf.float32)
x_test=tf.cast(x_test,tf.float32)
x_train=x_train/255.
x_test=x_test/255.
train_iter=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
test_iter=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)

# 定义输入，输出，隐藏层大小；初始化权重W，偏差b
num_inputs,num_outputs,num_hiddens=784,10,256
W1=tf.Variable(tf.random.normal(shape=(num_inputs,num_hiddens),mean=0.0,stddev=0.01,dtype=tf.float32))
b1=tf.Variable(tf.zeros(num_hiddens,dtype=tf.float32))
W2=tf.Variable(tf.random.normal(shape=(num_hiddens,num_outputs),mean=0.0,stddev=0.01,dtype=tf.float32))
b2=tf.Variable(tf.random.normal([num_outputs],stddev=0.1))

# 定义relu激活函数
def relu(X):
    return tf.math.maximum(X,0)
# 定义网络结构，返回softmax的分类概率结果
def net(X):
    X=tf.reshape(X,shape=[-1,num_inputs])
    h=relu(X)
    return tf.math.softmax(tf.matmul(h,W2)+b2)
# 定义损失函数
def loss(y_hat,y_true):
    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)

# 训练模型，定义参数
num_epochs,lr=5,0.1
params=[W1,b1,W2,b2]

# 采用上一个文件中的方法直接训练
train_che3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)




