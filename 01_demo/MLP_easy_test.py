'''
双层网络采用Relu激活函数对fashion_mnist分类
采用keras自带的方法简洁实现
'''
import tensorflow as tf
import sys
import tensorflow.keras as ks
from tensorflow.keras.datasets.fashion_mnist import load_data
# 定义网络结构和激活函数
model = ks.models.Sequential([
    ks.layers.Flatten(input_shape=(28,28)),
    ks.layers.Dense(256,activation='relu'),
    ks.layers.Dense(10,activation='softmax')
])
# 获取数据集
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
#编译网络模型
model.compile(optimizer=ks.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型，得到准确率
model.fit(x_train,y_train,batch_size=256,epochs=5,validation_data=(x_test,y_test),validation_freq=1)


