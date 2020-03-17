# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:13:06 2019

@author: HPH
"""

import os, shutil

original_dataset_dir = 'E:/BaiduNetdiskDownload/kaggle/train'
base_dir = 'E:/Project/Spyder-py3/cats_and_dogs_small'
# os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')  # 训练集路径
validation_dir = os.path.join(base_dir, 'validation')  # 验证集路径
test_dir = os.path.join(base_dir, 'test')  # 测试集路径
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

# 训练集，验证集，测试集下猫和狗的路径
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

# 复制图片
fnames = ['cat.{}.jpg'.format(i) for i in range(0, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)  # 复制
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(0, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))



from keras import models, layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # 卷积层
model.add(layers.MaxPool2D((2, 2)))  # 池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # 添加dropout降低过拟合
model.add(layers.Dense(512, activation='relu'))  # 密集连接层
model.add(layers.Dense(1, activation='sigmoid'))  # 二分类，sigmoid
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])


train_datagen = ImageDataGenerator(
        rescale=1/255,   # 将像素值缩小255倍
        rotation_range=40,  # 0-180角度值，图像随机旋转的范围
        width_shift_range=0.2,  # 比例，图像随机左右平移的比例
        height_shift_range=0.2,  # 图像随机上下平移的比例
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True,  # 随机将图片的一半水平翻转
        ) 
test_datagen = ImageDataGenerator(rescale=1/255) # 测试集不能使用数据增强

# 将图片转换为（150， 150, 3）的张量，返回值为生成器
train_generator = train_datagen.flow_from_directory(
        directory=train_dir,   # 目标路径
        target_size=(150, 150),   # 生成的图片大小
        batch_size=32,   # 每次返回20个数据
        class_mode='binary'
        )
validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
        )

history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,  # batch_size
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50
        )

# 保存模型
model.save('cats_and_dogs_small_2.h5')

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
# 训练集上损失越来越小，精度越来越高。
# 验证集上精度在第五轮后稳定在70%-72%之间
# 结论：随着训练次数的增加，模型开始出现过拟合



# =============================================================================
# 使用数据增强来降低过拟合
# 数据增强是在现有的训练数据上利用多种能生成可信图像的随机变换来增加样本
# 目的是在训练中不会两次查看到相同的数据，使模型学习到更多的内容拥有良好的泛化能力。
# datagen = ImageDataGenerator(
#         rotation_range=40,  # 0-180角度值，图像随机旋转的范围
#         width_shift_range=0.2,  # 比例，图像随机左右平移的比例
#         height_shift_range=0.2,  # 图像随机上下平移的比例
#         shear_range=0.2,  # 随机错切变换的角度
#         zoom_range=0.2,  # 随机缩放的范围
#         horizontal_flip=True,  # 随机将图片的一半水平翻转
#         fill_mode='nearest'  # 填充新像素的方法
#         )
# 
# from keras.preprocessing import image
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# img_path = fnames[3]
# img = image.load_img(img_path, target_size=(150, 150))  # 读取图片并调整大小
# x = image.img_to_array(img)
# x = x.reshape((1, ) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i == 4: break
# plt.show()
# =============================================================================