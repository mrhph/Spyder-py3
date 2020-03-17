# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:26:16 2019

@author: HPH
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
from keras import models

model = load_model('cats_and_dogs_small_2.h5')  # 导入模型

# 可视化中间激活。对于给的输入，展示网络中各个卷积层和池化层输出的特征图

# 加载图片，并将图片转换为4D张量。
img_path = 'E:\\Project\\Spyder-py3\\cats_and_dogs_small\\test\\cats\\cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

# 用输入张量和输出张量列表定义模型
layer_output = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, output=layer_output)

activations = activation_model.predict(img_tensor)

# plt.imshow(activations[0][0, : , : , 4], cmap='viridis')
# plt.show()

layer_names = [layer.name for layer in model.layers]
images_per_row = 16  # 每行存放的特征图个数

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]  # 特征图个数
    size = layer_activation.shape[1]  # 特征图的尺寸
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, size * images_per_row))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig('./cat_feature_imgs/' + layer_name + '.png')



# 可视化卷积神经网络的过滤器
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

def deprocess_image(x):
    x -= x.mean()  # 平均值
    x /= (x.std() + 1e-5)  # 方差
    x *= 0.1  # 标准差为0.1
    x += 0.5
    x = np.clip(x, 0, 1) # 限制数组的值在a_min和a_max之间
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(name=layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])  # Mean of tensor,return tensor
    
    # gradients(loss, variables), 返回veriables在loss上得梯度， return tensor
    grads = K.gradients(loss, model.input)[0]  # 获取损失相对于输入的梯度
    # 梯度标准化，除以L2范数（张量中所有值的平方和的平均值的开方）
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # 标准化梯度
    
    # K.function(inputs, outputs, updates=None, **kwargs)
    # 将一个numpy张量列表转换为两个numpy张量组成的列表
    iterate = K.function([model.input], [loss, grads])
    
    input_image_data = np.random.random((1, size, size, 3)) * 20 + 128.  # 定义随机初始输入
    # 进行梯度上升
    step = 1
    for i in range(40):
        value_loss, value_grads = iterate([input_image_data])
        input_image_data += value_grads * step
    img = input_image_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

layer_name = 'block1_conv1'
size = 64
margin = 5  # 边框的像素大小
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, j + (i * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()



# 可视化类激活的热力图
# 可以了解图像的哪一部分放网络做出了最终的决策
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


model = VGG16(weights='imagenet')  # 加载模型

# 加载图片，并将图片转换为numpy数组，最后用vgg16预处理numpy数组
img = image.load_img('creative_commons_elephant.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # 添加一个轴
x = preprocess_input(img)  # 预处理

preds = model.predict(x)  # 预测, return tensor, shape=(1, 1000)
print(decode_predictions(preds, top=3))  # 前三个权重最大的分类
# np.argmax(preds)  # 386, 最大值的索引

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')  # 最后一个卷积层

# K.gradients 计算梯度，返回值为包含梯度张量的列表。
# 最后一个卷积层在african_elephant_output上得梯度， (?, 14, 14 ,512)
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# 梯度张量的均值,(512, )
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
# 逐个比较heatmap和0的大小，取较大值
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.imshow(heatmap)
plt.show()

import cv2
img = cv2.imread('creative_commons_elephant.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap + 0.4 + img
cv2.imwrite('elephant_cam.jpg', superimposed_img)














