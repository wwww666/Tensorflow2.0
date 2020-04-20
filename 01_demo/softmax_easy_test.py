
'''
实现fashion_mnist简洁版，使用keras自带的框架
'''
import tensorflow as tf
import tensorflow.keras as ks

# 读取数据
fashion_mnist=ks.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
# 对图片做归一化处理，如果不做归一化测试准确率会下降5-6%个百分点
x_train=x_train/255.
x_test=x_test/255.

#设定模型，平铺图片加softmax层预测
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(28,28)),
    ks.layers.Dense(10,activation=tf.nn.softmax)
])
# 设定loss类型、优化器类型以及学习率
loss='sparse_categorical_crossentropy'
optimizer=ks.optimizers.SGD(0.1)
# 加载模型
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy','mse'])
# 开始训练
model.fit(x_train,y_train,batch_size=256,epochs=4)

# 查看在测试数据集上的准确率
test_loss,test_acc,_=model.evaluate(x_test,y_test)
pred=model.predict(x_test)
print('Test acc:',test_acc)

