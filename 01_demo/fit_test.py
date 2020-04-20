'''
模拟多项式函数的拟合过程
'''
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# 自定义训练集、测试集、权重和偏量
n_train,n_test,true_w,true_b=100,100,[1.2,-3.4,5.6],5
features=tf.random.normal(shape=(n_train+n_test,1))
poly_features=tf.concat([features,tf.pow(features,2),tf.pow(features,3)],axis=1)
print(poly_features.shape)
labels=(true_w[0]*poly_features[:,0]+true_w[1]*poly_features[:,1]+true_w[2]*poly_features[:,2]+true_b)
print(labels.shape)
labels+=tf.random.normal(shape=labels.shape,mean=0.0,stddev=0.1)
# 定义图形样式、大小
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(5,3)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize
# 自定义作图函数
def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend=None,figsize=(5,3)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)
    plt.show()


# 定义训练函数，实验拟合效果
num_epochs,loss=100,tf.losses.MeanSquaredError()
def fit_and_plot(train_features,test_features,train_labels,test_labels):
    net=ks.Sequential()
    net.add(ks.layers.Dense(1))
    batch_size=min(10,train_labels.shape[0])
    train_iter = tf.data.Dataset.from_tensor_slices((train_features,train_labels)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((test_features,test_labels)).batch(batch_size)
    optimizer =ks.optimizers.SGD(0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                l = loss(y, net(X))

            grads = tape.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

        train_ls.append(loss(train_labels, net(train_features)).numpy().mean())
        test_ls.append(loss(test_labels, net(test_features)).numpy().mean())
    print(train_ls)
    print(test_ls)
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',range(1,num_epochs+1),test_ls,['train','test'])
    print('weight:',net.get_weights()[0],
          '\nbias:',net.get_weights()[1])

# fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
#              labels[:n_train], labels[n_train:])

fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])