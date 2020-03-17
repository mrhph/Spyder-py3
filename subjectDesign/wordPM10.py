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
    # 查看文件的前五行和后两行
    df = pandas.read_csv(inputFile('.csv'), encoding="gbk")
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
    # 选择指定的列数据，并导入到新的txt文件
    df = pandas.read_csv(inputFile('.csv'), encoding="gbk")
    new_df = pandas.DataFrame()  # 构建新数据集
    col = ['Region', 'Country', 'City/station', 'PM10', 'PM10 Year']
    new_df[col] = df[col]
    new_df.dropna(inplace=True)  # 清除空值
    print('#新数据集构建完成！开始保存文件...')
    out_path = outputFile('.txt')
    new_df.to_csv(out_path, index=0, sep=' ', encoding="gbk")  # 保存新数据，index=0 不保存索引, sep分隔符
    print('=>文件保存成功，{}'.format(out_path))
    
    
    
def toExcel():
    # 选择三列，保存为Excel文件
    df = pandas.read_csv(inputFile('.csv'), encoding="gbk")
    new_df = pandas.DataFrame()  # 构建新数据集
    col = ['City/station', 'PM10', 'PM10 Year']
    new_df[col] = df[col]
    new_df.dropna(inplace=True)  # 清除空值
    print('#新数据集构建完成！开始保存文件...')
    out_path = outputFile('.xlsx')
    new_df.to_excel(out_path, index=0, encoding="gbk")  # 保存新数据，index=0不保存索引
    print('=>文件保存成功，{}'.format(out_path))
    

def disperse():
    # 任务四，离散化处理，并将离散值绘制为饼图
    df = pandas.read_excel(inputFile('.xlsx'), encoding="gbk")
    category = [0, 50, 100, 150, 200]
    labels = ['One', 'Two', 'Three', 'Four']
    dis = pandas.cut(df['PM10'], category, labels=labels)  # 离散化
    nums = dis.value_counts()  # 统计离散数值的数量
    # 饼图
    plt.figure(figsize=(6,6))
    plt.pie(nums, labels=labels, autopct='%1.1f%%', explode=[0.01, 0.01, 0.01, 0.01])
    plt.title('PM10 离散化结果统计比例饼图')
    # plt.show()
    print('#饼状图绘制完毕，保存png文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=300)
    print('=>保存完毕（{}）'.format(out_path))
    

def main():
    # 菜单
    # 指定相应的编号，执行相应的函数
    while True:
        print('\n***********************************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件并查看前五和后二行')
        print('2.读取csv文件选择指定列导出到新txt文件')
        print('3.读取csv文件选择指定列并导出到Excel文件')
        print('4.读取csv文件对PM10进行离散化处理并绘制柱状图和饼图')
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
            toExcel()  # 任务三
        elif num == 4:  
            disperse()  # 任务四
        elif num == 0:
            break  # 退出
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()