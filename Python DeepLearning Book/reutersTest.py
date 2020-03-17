# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:12:42 2019

@author: HPH
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import reuters
from keras import layers, models

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# one-hot编码，将序列转换为指定维度的向量
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)  # 将训练数据向量化
x_test = vectorize_sequences(test_data)  # 将测试数据向量化

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

one_hot_train_labels = to_one_hot(train_labels)  # 将训练标签one_hot
one_hot_test_labels = to_one_hot(test_labels)  # 将测试标签one_hot
# keras内置实现了one_hot方法
# from keras.utils.np_utils import to_categorical
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# 将训练集划分为训练数据和验证数据
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 定义模型，隐藏层
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# 编译
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练
history = model.fit(x=partial_x_train, 
          y=partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_val, y_val)
          )


# 获取模型在训练集和验证集上得损失和精度
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

eqochs = range(1, len(loss)+1)

# 绘制训练集与验证集的损失值折线图
plt.plot(eqochs, loss, 'bo', label='Training loss')
plt.plot(eqochs, val_loss, 'b', label='Validaction loss')
plt.title('Training and validaction loss')
plt.xlabel('Eqochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练集与验证集的精度折线图
plt.plot(eqochs, acc, 'bo', label='Training acc')
plt.plot(eqochs, val_acc, 'b', label='Validaction acc')
plt.title('Training and validaction accuracy')
plt.xlabel('Eqochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()