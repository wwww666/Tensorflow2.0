'''
加载fashion_mnist，可视化展示
'''
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

feature,label=x_train[0],y_train[0]
# print(feature,label)
# 将标签换成文字表示
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pollover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]
# 展示图片
def show_fashion_mnist(images,labels):
    _,figs=plt.subplots(1,len(images),figsize=(12,12))
    for f ,img,lbl in zip(figs,images,labels):
        f.imshow(img.reshape((28,28)))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
#提取10张
X,y = [],[]
for i in range(10):
    X.append(x_train[i])
    y.append(y_train[i])
show_fashion_mnist(X,get_fashion_mnist_labels(y))

batch_size=256
if sys.platform.startswith('win'):
    num_works=0
else:
    num_works=4
train_iter=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)

start=time.time()
for X,y in train_iter:
    continue
print('%.2f sec'%(time.time()-start))