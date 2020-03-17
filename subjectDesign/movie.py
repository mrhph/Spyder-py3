# -*- coding: utf-8 -*-

import os
import pandas, matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# 设置matplotlib可以显示中文
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False 


def inputFile(file_type):
    # 接收并处理输入文件
    # file_type:输入文件类型
    print('#请输入读取文件(如:c:\\test.csv),文件类型:{}'.format(file_type))
    while True:
        path = input('>>> ')
        if not os.path.isfile(path):  # 是否是文件
            print('*ERROR:该文件不存在')
            continue
        if os.path.splitext(path)[1] != file_type:  # 将path俺路径和文件后缀分割，返回二元组
            print('*ERROR:文件类型不正确')
            continue
        return path


def outputFile(file_type):
    # 接收并处理数出文件路径
    print('#请输入保存文件(如:c:\\test.csv),保存文件类型:{}'.format(file_type))
    while True:
        path = input('>>> ')
        file_path = os.path.split(path)[0]  # 路径
        if not os.path.isdir(file_path):  # 路径是否存在
            print('*ERROR:路径不存在')
            continue
        if os.path.splitext(path)[1] != file_type:
            print('*ERROR:文件类型不正确')
            continue
        return path
    

def lookHeadAndTail():
    # 查看前三行和后两行
    df = pandas.read_csv(inputFile('.csv'))
    print('########################################')
    print('前三行:')
    for i in range(3):
        print(list(df.iloc[i]))
    print('后二行:')
    for i in range(2):
        print(list(df.tail(3).iloc[i]))
    print('#######################################')
        
        
def newDf():
    # 选择指定列，处理缺失值
    df = pandas.read_csv(inputFile('.csv'))
    new_df = pandas.DataFrame()  # 构建新数据集
    col = ['budget', 'id', 'release_date', 'revenue', 'title', 'original_language']
    new_df[col] = df[col]
    new_df.dropna(inplace=True)  # 清除空值
    print('#新数据集构建完成！开始保存文件...')
    out_path = outputFile('.csv')
    new_df.to_csv(out_path, index=0)  # index=0 不保存索引
    print('=>文件保存成功，{}'.format(out_path))
    
    
    
def choiceEn():
    # 选择original_language为en并导出txt文件
    df = pandas.read_csv(inputFile('.csv'))
    new_df = df[df['original_language']=='en']  # en
    print('#选择original_language为en操作完成！开始保存文件...')
    out_path = outputFile('.txt')
    new_df.to_csv(out_path, index=0)
    print('=>文件保存成功，{}'.format(out_path))
    
    
def profit():
    # 计算profit添加到数据集并保存到csv文件
    df = pandas.read_csv(inputFile('.txt'))
    df['profit'] = df['revenue'] - df['budget']  # 计算profit并赋值新列
    print('#利润计算完成！开始保存文件...')
    out_path = outputFile('.csv')
    df.to_csv(out_path, index=0)
    print('=>文件保存成功，{}'.format(out_path))
    

def plot():
    # 绘制关于Budget、 revenue、 Profit 的图形
    df = pandas.read_csv(inputFile('.csv'))
    cols = ['budget', 'revenue', 'profit']
    colors = ['red', 'green', 'blue']
    titles = ['预算', '收入', '利润']
    for i in range(3):
        n_df = df.sort_values(by=cols[i], ascending=False)  # 排序，降序
        temp = [n_df[cols[i]].iloc[j] for j in range(0, len(n_df), 200)]  # 步长200取值
        x_ticks = [n_df['title'].iloc[j] for j in range(0, len(n_df), 200)]  # 取标题
        plt.figure(figsize=(8,6))  # 指定图的大小 ，(width, height)
        plt.plot(range(len(temp)), temp, color=colors[i], label=cols[i])  # 画图
        plt.xticks(range(len(temp)), x_ticks, rotation=90)  # rotation刻度旋转角度
        plt.legend()  # 显示图例
        plt.xlabel('电影')  # x轴标签
        plt.ylabel('数值')  # y轴标签
        plt.title('1990-2018 年华语电影{}折线图'.format(titles[i]))  #标题
        print('#{}折线图绘制完成！开始保存文件...'.format(cols[i]))
        out_path = outputFile('.png')
        plt.savefig(out_path, dpi=400)
        # plt.show()
        print('=>文件保存成功，{}'.format(out_path))
    

def main():
    # 菜单
    while True:
        print('\n***********************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件并查看前三和后二行')
        print('2.读取csv文件删除缺失值和列并导出到新csv文件')
        print('3.读取csv文件选取华语电影并导出到txt文件')
        print('4.读取txt文件计算利润并保存到csv文件')
        print('5.读取csv文件绘制关于budget,revenue,profit的折线图')
        print('0.退出')
        print('*************************************************')
        num = input('>>> ')
        try:
            num = int(num)
        except:
            print('*ERROR:请输入数字编号')
        if num == 1:
            lookHeadAndTail()
        elif num == 2:
            newDf()
        elif num == 3:
            choiceEn()
        elif num == 4:
            profit()
        elif num == 5:
            plot()
        elif num == 0:
            break
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()