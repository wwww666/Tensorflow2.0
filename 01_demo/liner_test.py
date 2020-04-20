import tensorflow as tf
from matplotlib import pyplot as plt
import random
# 自定义数据集
num_inputs = 2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2

features=tf.random.normal((num_examples,num_inputs),stddev=1)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=tf.random.normal(labels.shape,stddev=0.01)
print(features[0],labels[0])


# 查看features和labels的散点图关系
def set_figsize(figsize=(5,3)):
    plt.rcParams['figure.figsize']=figsize
set_figsize()
plt.scatter(features[:,1],labels,1)
plt.show()

# 定义batch，按批次大小读取数据
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
         j=indices[i:min(i+batch_size,num_examples)]  #批次读取，min（）是将不足一个batch的读取进来当做一个批次
         yield tf.gather(features,axis=0,indices=j),tf.gather(labels,axis=0,indices=j) #迭代器获取每个batch

# 按照每批次10组数据读取
batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break
# 定义初始化权重
w=tf.Variable(tf.random.normal((num_inputs,1),stddev=0.01))
b=tf.Variable(tf.zeros(1,))

# 线性计算
def linreg(X,w,b):
    return tf.matmul(X,w)+b

# 定义loss，使用平方损失
def squared_loss(y_hat,y):
    return (y_hat-tf.reshape(y,y_hat.shape))**2/2

# 定义优化器
def sgd(params,lr,batch_size,grads):
    for i,param in enumerate(params):
        param.assign_sub(lr*grads[i]/batch_size)

# 定于训练参数
lr=0.03
num_epochs=3
net=linreg
loss=squared_loss
# 训练数据
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with tf.GradientTape() as t:
            t.watch([w,b])
            l=loss(net(X,w,b),y)
        grads=t.gradient(l,[w,b])
        sgd([w,b],lr,batch_size,grads)
    train_l = loss(net(features,w,b),labels)
    print('epoch%d,loss%f'%(epoch+1,tf.reduce_mean(train_l)))

# 比较学习到的的w,b与真实定义的w,b
print(true_w,w)
print(true_b,b)
