# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:02:23 2019

@author: HPH
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb  # 导入数据
from keras import models, layers  # 导入模型，层


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # 加载数据

# one-hot编码，将序列转换为指定维度的向量
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)  # 将训练数据向量化
x_test = vectorize_sequences(test_data)  # 将测试数据向量化

# 标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 定义模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
# 优化器：rmsprop，损失函数binary_acrossentropy（二元交叉熵）, 监控指标：accuracy
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# 训练集上取前1000个作为验证集
x_val = x_train[:10000]
partial_x_val = x_train[10000:]
y_val = y_train[:10000]
partial_y_val = y_train[10000:]

# 训练数据
history = model.fit(x=partial_x_val,
                    y=partial_y_val,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(loss)+1)

# 绘制训练损失与验证损失
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度与验证精度
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# 为防止过拟合，对权重系数添加L2正则化。
from keras import regularizers
model2 = models.Sequential()
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history2 = model2.fit(x=partial_x_val,
                      y=partial_y_val,
                      epochs=20,
                      batch_size=512,
                      validation_data=(x_val, y_val))

epochs_num = range(1, 21)
plt.plot(epochs_num, history.history['loss'], 'bo', label='Original model loss')
plt.plot(epochs_num, history2.history['loss'], 'b+', label='L2-regularized model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs_num, history.history['val_loss'], 'bo', label='Original model validation loss')
plt.plot(epochs_num, history2.history['val_loss'], 'b+', label='L2-regularized model validation loss')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()
# 结论
# 在添加L2正则化上的模型的验证损失上什较慢。表示L2正则化的模型比原模型更不容易过拟合


# dropout正则化
model3 = models.Sequential()
model3.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(16, activation='relu'))
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(1, activation='sigmoid'))
model3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history3 = model3.fit(partial_x_val, 
           partial_y_val, 
           epochs=20, 
           batch_size=512, 
           validation_data=(x_val, y_val))

plt.plot(epochs_num, history.history['loss'], 'bo', label='Original model loss')
plt.plot(epochs_num, history3.history['loss'], 'b+', label='Dropout model validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs_num, history.history['val_loss'], 'bo', label='Original model loss')
plt.plot(epochs_num, history3.history['val_loss'], 'b+', label='Dropout model validation loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()