'''
手动定义网络模型实现softmax对fashion_mnist分类
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
# 定义批次大小，读取数据集
batch_size=256
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
# print(x_test.dtype,y_test.dtype)
# 归一化数据集，减小训练误差
x_train=tf.cast(x_train,dtype=tf.float32)/255
x_test=tf.cast(x_test,dtype=tf.float32)/255
# 采用tf自带的分割方法，按批次大小返回数据集
train_iter = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)
# 定义初始化权重矩阵及偏秩；定义输入输出大小
num_inputs=784
num_outputs=10
W=tf.Variable(tf.random.normal(shape=(num_inputs,num_outputs),mean=0,stddev=0.01,dtype=tf.float32))
b=tf.Variable(tf.zeros(num_outputs,dtype=tf.float32))
# X=tf.constant([[1,2,3],[3,4,5]])
# print(tf.reduce_sum(X,axis=-1,keepdims=True))

# softmax计算公式
def softmax(logits,axis=-1):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits),axis,keepdims=True)
# X=tf.random.normal(shape=(2,5))
# X_prob=softmax(X)
# print(X)
# print(X_prob)
# print(tf.reduce_sum(X_prob,axis=1))

# 定义网络结构，返回softmax计算结果
def net(X):
    logits=tf.matmul(tf.reshape(X,shape=(-1,W.shape[0])),W)+b
    return softmax(logits)

# 测试，定义预测数组及真实值下标数组
y_hat=np.array([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y=np.array([0,2],dtype=np.int32)

# print(tf.boolean_mask(y_hat,tf.one_hot(y,depth=3)))

# 计算loss，将指标转换并做one_hot处理
def cross_entropy(y_hat,y):
    y=tf.cast(tf.reshape(y,shape=[-1,1]),dtype=tf.int32)
    y=tf.one_hot(y,depth=y_hat.shape[-1])
    y=tf.cast(tf.reshape(y,shape=[-1,y_hat.shape[-1]]),dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat,y)+1e-8)

# 计算准确率，采用上面定义的测试数据
def accuracy(y_hat,y):
    return np.mean(tf.argmax(y_hat,axis=1).numpy()==y)
# print(np.sum([2,2]==[2,2]))
# print(tf.argmax(y_hat,axis=1))
print(accuracy(y_hat,y))

# 计算批次的准确率
def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for batch,(X,y) in enumerate(data_iter):
        y=tf.cast(y,dtype=tf.int64).numpy()
        acc_sum+=np.sum(tf.cast(tf.argmax(net(X),axis=1),dtype=tf.int64).numpy()==y)
        n+=y.shape[0]
    return acc_sum/n
# print(evaluate_accuracy(test_iter,net))

# 定义迭代次数，定义初始化学习率
num_epochs,lr=5,0.1

# 训练网络
def train_che3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,trainer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            with tf.GradientTape() as tape:
                y_hat=net(X)
                l=tf.reduce_sum(loss(y_hat,y))
            gards=tape.gradient(l,params)
            if trainer is None:
                for i,param in enumerate(params):
                    param.assign_sub(lr*gards[i]/batch_size)
            else:
                trainer.apply_gradients(zip([gard/batch_size for gard in gards],params))

            y=tf.cast(y,dtype=tf.float32)
            train_l_sum+=l.numpy()
            train_acc_sum+=tf.reduce_sum(tf.cast(tf.argmax(y_hat,axis=1).numpy()==tf.cast(y,dtype=tf.int64).numpy(),dtype=tf.int64)).numpy()
            n+=y.shape[0]
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch%d,loss%.4f,train_acc%.3f,test_acc%.3f'%(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))

# 定义优化器
trainer=SGD(lr)
train_che3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[W,b],lr,trainer)

# 迭代读取训练数据集（图片、标签）
X,y = iter(test_iter).next()

# 将训练数据集标签转换为文字表示
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pollover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 可视化预测结果，预测标签与真实标签对比展示
def show_fashion_mnist(images,labels):
    _,figs=plt.subplots(1,len(images),figsize=(28,28))
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(tf.reshape(img,shape=(28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
true_labels=get_fashion_mnist_labels(y.numpy())
pred_labels=get_fashion_mnist_labels(tf.argmax(net(X),axis=1).numpy())
titles=[true+'\n'+pred for true,pred in zip(true_labels,pred_labels)]

# 取前10个图片和标签
show_fashion_mnist(X[:9],titles[:9])
