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
    # 查看文件的前五行和后后两行
    df = pandas.read_csv(inputFile('.csv'))
    df.dropna(inplace=True)  # 清除空值
    print('########################################')
    print('前五行:')
    for i in range(5):
        print(list(df.iloc[i]))
    print('后二行:')
    for i in range(2):
        print(list(df.tail(2).iloc[i]))
    print('#######################################')
        
        
def newDf():
    # 选择指定的六列数据，并导入到新的txt文件
    df = pandas.read_csv(inputFile('.csv'))
    new_df = pandas.DataFrame()  # 构建新数据集
    col = ['Id', 'KitchenQual', 'LotArea', 'OverallCond', 'YrSold', 'SalePrice']
    new_df[col] = df[col]
    new_df.dropna(inplace=True)  # 清除空值
    print('#新数据集构建完成！开始保存文件...')
    out_path = outputFile('.txt')
    new_df.to_csv(out_path, index=0, sep=' ')  # 保存新数据，index=0 不保存索引, sep分隔符
    print('=>文件保存成功，{}'.format(out_path))
    
    
    
def unitPrice():
    # 计算unitPrice
    df = pandas.read_csv(inputFile('.txt'), sep=' ')
    df['unitPrice'] = df['SalePrice'] / df['LotArea']  # 计算profit并赋值新列
    print('#单位价格计算完成！开始保存文件...')
    out_path = outputFile('.xlsx')
    df.to_excel(out_path, index=0)
    print('=>文件保存成功，{}'.format(out_path))
    

def unitPriceMean():
    # 分组unitPrice的均值，绘图
    df = pandas.read_excel(inputFile('.xlsx'))
    # groupby:分组, mean:求均值, sort_values: 值排序， ascending=False: 降序
    group_unit_price_mean = df.groupby('KitchenQual')['unitPrice'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,6))  # 指定图的大小, (width, height)
    plt.bar(group_unit_price_mean.index, group_unit_price_mean, width=0.5, label='分组单位均价')  #柱状图
    plt.legend()  # 显示图例
    # plt.yticks(range(0, 25, 5), ['Poor', 'Fair', 'Typical', 'Good', 'Excellent'])
    plt.xlabel('KitchenQual')
    plt.ylabel('unitPrice')
    plt.title('分组unitPrice均价（降序）柱状图')
    print('#柱状图绘制完成！开始保存文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=300)
    # plt.show()
    print('=>文件保存成功，{}'.format(out_path))
    

def overallCondMean():
    # 分组overallCond的均值，绘图
    df = pandas.read_excel(inputFile('.xlsx'))
    group_overall_cond_mean = df.groupby('KitchenQual')['OverallCond'].mean().sort_values()
    plt.figure(figsize=(8,6))
    plt.bar(group_overall_cond_mean.index, group_overall_cond_mean, width=0.5, label='分组单位均价')
    plt.legend()
    plt.xlabel('KitchenQual')
    plt.ylabel('OverallCond')
    plt.title('房子整体状况评估（升序）柱状图')
    print('#柱状图绘制完成！开始保存文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=300)
    # plt.show()
    print('=>文件保存成功，{}'.format(out_path))
    

def main():
    # 菜单
    # 指定相应的编号，执行相应的函数
    while True:
        print('\n***********************************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件并查看前五和后二行')
        print('2.读取csv文件选择指定列导出到新txt文件')
        print('3.读取txt文件计算unitPrice并导出到xlsx文件')
        print('4.读取xlsx文件利用列KitchenQual分组计算unitPrice均值并绘制柱状图')
        print('5.读取xlsx文件利用列KitchenQual分组计算OverallCond均值并绘制柱状图')
        print('0.退出')
        print('*************************************************************')
        num = input('>>> ')
        try:
            num = int(num)
        except:
            print('*ERROR:请输入数字编号')
        if num == 1:
            lookHeadAndTail()  # 任务一
        elif num == 2:
            newDf()  #任务二
        elif num == 3:
            unitPrice()   # 任务三
        elif num == 4:  
            unitPriceMean()  # 任务四
        elif num == 5:  
            overallCondMean()  # 任务五
        elif num == 0:
            break  # 退出
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()