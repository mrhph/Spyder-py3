# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:29:10 2019

@author: HPH
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras import models, layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化，对输入数据的每个特征先减去特征平均值再除以标准差
mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std
# 测试集数据标准化的均值和标准差都在训练集上的，不能在测试集上计算任何数据
test_data -= mean
test_data /= std


# 构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # 均方误差MSE， 平均绝对值误差MAE
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K折交叉验证
# 将数据划分为K个分区，实例化K个相同的模型
# 每个模型上选取K-1个分区作为训练数据，剩下的一个分区做验证数据
# 模型的验证分数为K个模型验证分数的平均值

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('Processing flod #', i)
    # 准备验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    # 准备训练数据
    # np.concatenate()连接张量
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()  # 构建模型
    # 训练模型， verbose = 0 # 静默模式
    history = model.fit(x=partial_train_data, y=partial_train_targets,
              validation_data=(val_data, val_targets),
              epochs=num_epochs, batch_size=1, verbose=1)
    # 记录每次迭代中的MAE（验证分数），为长度为epochs的列表
    mae_history = history.history['val_mean_absolute_error']
    print(mae_history)
    all_mae_histories.append(mae_history)  # K折训练中的所有MAE（验证分数）
    

# 计算K折中所有迭代验证分数的平均值，该平均值即为模型最终的验证分数
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# 绘制验证分数折线图
# plt.plot(range(1, num_epochs + 1), average_mae_history, 'b-')
# plt.title('Validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

# 删除前十个数据点
# 将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 80次迭代后开始出现过拟合
# 最终训练
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_sorce, test_mae_sorce = model.evaluate(test_data, test_targets)