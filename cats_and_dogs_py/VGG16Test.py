# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:10:23 2019

@author: HPH
"""

# 使用训练好的VGG16架构
import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

base_dir = 'E:/Project/Spyder-py3/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')  # 训练集路径
validation_dir = os.path.join(base_dir, 'validation')  # 验证集路径
test_dir = os.path.join(base_dir, 'test')  # 测试集路径

# 取得VGG16的卷积基
conv_base = VGG16(weights='imagenet',  # 初始化权重的检查点
                  include_top=False,   # 是否包含密集连接层
                  input_shape=(150, 150, 3))  # 输入数据的形状

datagen = ImageDataGenerator(1/255)
batch_size = 20

# =============================================================================
# # 使用VGG16卷积基的两种方式
# # 方式一
# # 将数据输入到VGG16的卷积基，输出numpy数组，将输出的数组作为独立的密集连接层的输入
# # 优点是训练速度快，计算代价小。没一个独立的图像只需运行一次卷积基。但是无法使用数据增强
# # 方式二
# # 在卷积基的顶部添加Dense层，以扩展已有的VGG16的卷积基。此方式可以使用数据增强，但是计代价高
# =============================================================================

# 方式一，添加独立的Dense层
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, ))
    generator = datagen.flow_from_directory(
            directory=directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 密集连接层只能接收一维张量的输入，将三维转换为一维，类似于Flatten层
train_features = train_features.reshape(2000, 4 * 4 * 512)
validation_features = validation_features.reshape(1000, 4 * 4 * 512)
test_features = test_features.reshape(1000, 4 * 4 * 512)

# 定义密集连接层
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(
        x = train_features,
        y = train_labels,
        batch_size=100,
        epochs=30,
        validation_data=(validation_features, validation_labels)
        )
# 提取训练结果并绘图
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)
# 绘图
plt.plot(epochs, acc, 'r-', label='Train accuracy')
plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'r-', label='Train loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 方式二，拓展已有的卷积基
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False  # 不修改卷积基中的权重
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(
        rescale=1/255,  # 尺寸缩小
        rotation_range=40,  # 旋转的角度范围
        height_shift_range=0.2,  # 上下平移的比例范围
        width_shift_range=0.2,   # z左右平移的比例范围
        shear_range=0.2,  # 错切变换的范围
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True,  # 随机将图片的一半翻转
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
        )
validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
# 提取训练结果并绘图
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)
# 绘图
plt.plot(epochs, acc, 'r-', label='Train accuracy')
plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'r-', label='Train loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 对方式二进行模型微调
# 将卷积基的顶部进行部分解冻，这里解冻最后的三个卷积层和池化层
# =============================================================================
# # 在已经训练好的基网络上添加自定义网络
# # 冻结基网络
# # 训练添加的自定义网络
# # 解冻一部分基网络
# # 联合训练解冻的基网络和添加的自定义网络
# =============================================================================
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 重新编译模型，使用学习率更小的优化器
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,  # 生成器每次返回20条数据，训练集共2000个，共返回100次
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)
# 提取训练结果并绘图
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)

# 指数移动平均值EMA, EMA_today = a * Price_today + (1 - a) * EMA_yeatarday
def smooth_curve(points, factor=0.8):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous * factor + point * (1 - factor))
        else:
            smooth_points.append(point)
    return smooth_points

# 绘图
plt.plot(epochs, smooth_curve(acc), 'r-', label='Train accuracy')
plt.plot(epochs, smooth_curve(val_acc), 'b-', label='Validation accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, smooth_curve(loss), 'r-', label='Train loss')
plt.plot(epochs, smooth_curve(val_loss), 'b-', label='Validation loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
        )
acc_test, loss_test = model.evaluate_generator(test_generator, steps=50)
print('acc_test: {},  loss_loss: {}'.format(acc_test, loss_test))


















