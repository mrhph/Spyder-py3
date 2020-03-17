# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:05:34 2019

@author: HPH
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False 

# 数学 math， 语文 chinese, 英语 english 
# 政治 politics， 历史 history， 地理 geography
# 物理 physics， 化学 chemistry， 生物 biology

subjects = ['语文', '数学', '英语', '政治', '历史', '地理', '物理', '化学', '生物']

one = pd.read_excel('./xiaowu/one.xlsx')
two = pd.read_excel('./xiaowu/two.xlsx')
average_one = np.mean(one, axis=0) # 均值
max_one = np.max(one, axis=0)  # 最大值
min_one = np.min(one, axis=0)  # 最小值
med_one = np.median(one.drop(['姓名'], axis=1), axis=0)  # 中值
result_average_one = (average_one * len(one) - max_one - min_one) / (len(one) - 2)
result_average_one = [round(result_average_one[i], 3) for i in subjects]  # 去掉最大和最小值的均值

average_two = np.mean(two, axis=0)
max_two = np.max(two, axis=0)
min_two = np.min(two, axis=0)
med_two = np.median(two.drop(['姓名'], axis=1), axis=0)
result_average_two = (average_two * len(two) - max_two - min_two) / (len(two) - 2)
result_average_two = [round(result_average_two[i], 3) for i in subjects]


def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), # put the detail data
                    xy=(rect.get_x() + rect.get_width() / 2, height), # get the center location.
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

def auto_text(rects):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')


fig, ax = plt.subplots(figsize=(12, 9))
bar1 = ax.bar(np.arange(9), result_average_one, width=0.4, alpha=0.8, label='第一次月考成绩')
bar2 = ax.bar(np.arange(9)+0.4, result_average_two, width=0.4, alpha=0.8, label='段考成绩')
plt.xticks(np.arange(9)+0.2, subjects)
auto_text(bar1)
auto_text(bar2)
plt.grid(ls='--', axis='y', alpha=0.5)  # 打开坐标网格， 绘制水平线
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
plt.legend()
plt.xlabel('学科')
plt.ylabel('成绩')
plt.yticks(range(0, 90, 10))
plt.title('高一年级十六班各科成绩平均值(去掉最高和最低分)柱状图')
plt.savefig('./xiaowu/高一年级十六班各科成绩平均值(去掉最高和最低分)柱状图.png')
plt.show()



def score_range(scores, full_mark=100):
    # 统计成绩区间
    count = [0, 0, 0, 0, 0]
    for score in scores:
        if 0 <= score < full_mark * 0.6:
            count[0] += 1
        elif full_mark * 0.6 <= score < full_mark * 0.7:
            count[1] += 1
        elif full_mark * 0.7 <= score < full_mark * 0.8:
            count[2] += 1
        elif full_mark * 0.8 <= score < full_mark * 0.9:
            count[3] += 1
        elif full_mark * 0.9 <= score <= full_mark:
            count[4] += 1
    # result = [round(i/len(scores), 3) for i in count]
    return count

labels1 = ['0-89','90-104','105-119','120-134','135-150']
labels2 = ['0-59','60-69','70-79','80-89','90-100']
explode = [0.01] * 5


plt.subplots(figsize=(12, 12))
i = 1
for subject in subjects:
    plt.subplot(3,3,i)
    sizes = score_range(one[subject])
    if subject in ['语文', '数学', '英语']:
        labels = labels1
    else:
        labels = labels2
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.3f%%')
    plt.title(subjects[i-1])
    i += 1
plt.suptitle('高一年级十六班第一次月考各科成绩区间分布饼图')
plt.savefig('./xiaowu/高一年级十六班第一次月考各科成绩区间分布饼图.png')
plt.show() 


plt.subplots(figsize=(12, 12))
plt.subplot(4, 1, 1)
i = 1
for subject in subjects:
    plt.subplot(3,3,i)
    sizes = score_range(two[subject])
    if subject in ['语文', '数学', '英语']:
        labels = labels1
    else:
        labels = labels2
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.3f%%')
    plt.title(subjects[i-1])
    i += 1
plt.suptitle('高一年级十六班段考各科成绩区间分布饼图')
plt.savefig('./xiaowu/高一年级十六班段考各科成绩区间分布饼图.png')
plt.show()