# -*- coding: utf-8 -*-

import os
import pandas, matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from collections import Counter


matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False 


def inputFile(file_type):
    print('#请输入读取文件(如:c:\\test.csv),文件类型:{}'.format(file_type))
    while True:
        path = input('>>> ')
        if not os.path.isfile(path):
            print('*ERROR:该文件不存在')
            continue
        if os.path.splitext(path)[1] != file_type:
            print('*ERROR:文件类型不正确')
            continue
        return path


def outputFile(file_type):
    print('#请输入保存文件(如:c:\\test.csv),保存文件类型:{}'.format(file_type))
    while True:
        path = input('>>> ')
        file_path = os.path.split(path)[0]
        if not os.path.isdir(file_path):
            print('*ERROR:路径不存在')
            continue
        if os.path.splitext(path)[1] != file_type:
            print('*ERROR:文件类型不正确')
            continue
        return path
    

def lookHeadAndTailThree():
    file = pandas.read_csv(inputFile('.csv'))
    print('########################################')
    print('前三行:')
    for i in range(3):
        print(list(file.iloc[i]))
    print('后二行:')
    for i in range(2):
        print(list(file.tail(3).iloc[i]))
    print('#######################################')
        
        
def dropNullAndRow():
    file = pandas.read_csv(inputFile('.csv'))
    new_file = file.dropna()  # 删除空值
    drop_col = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
    new_file.drop(drop_col, axis=1, inplace=True)  # 删除列，inplace=True在原数据上操作
    print('#空值及列删除操作成功！开始保存文件...')
    out_path = outputFile('.csv')
    new_file.to_csv(out_path, index=0)  # index=0 不保存索引
    print('=>文件保存成功，{}'.format(out_path))
    
    
    
def toTxt():
    df = pandas.read_csv(inputFile('.csv'))
    df = df[df['pm2.5'].gt(300)]  # pm2.5大于300
    print('#操作成功！开始保存文件...')
    out_path = outputFile('.txt')
    df.to_csv(out_path, index=0)
    print('=>文件保存成功，{}'.format(out_path))
    
    
def toExcel():
    df = pandas.read_csv(inputFile('.txt'))
    print('#操作成功！开始保存文件...')
    out_path = outputFile('.xlsx')
    df.to_excel(out_path, index=0)
    print('=>文件保存成功，{}'.format(out_path))
    

def findMax(d):
    d = dict(d)
    key, val = -1, -1
    for k, v in d.items():
        if v > val:
            key, val = k, v
    return key, val


def plot():
    df = pandas.read_csv(inputFile('.txt'))
    m_k, m_v = findMax(Counter(df['month']))  # Counter
    d_k, d_v = findMax(Counter(df['day']))
    h_k, h_v = findMax(Counter(df['hour']))
    plt.figure(figsize=(8,6))
    plt.bar(1, m_v, color='red',label='month')
    plt.bar(3, d_v, color='green', label='day')
    plt.bar(5, h_v, color='blue', label='hour')
    plt.legend()
    plt.xlabel('month, day, hour')
    plt.ylabel('频次')
    plt.title('出现次数最多的month, day, hour的频次直方图')
    print('#操作成功！开始保存文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=400)
    # plt.show()
    print('=>文件保存成功，{}'.format(out_path))
    

def main():
    while True:
        print('\n***********************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件并查看前三和后二行')
        print('2.读取csv文件删除缺失值和列并导出到新csv文件')
        print('3.读取csv文件选取pm2.5大于300的数据并导出到txt文件')
        print('4.读取txt文件并保存为xlsx文件')
        print('5.读取txt文件绘制关于month,day,hour的直方图')
        print('0.退出')
        print('*************************************************')
        num = input('>>> ')
        try:
            num = int(num)
        except:
            print('*ERROR:请输入数字编号')
        if num == 1:
            lookHeadAndTailThree()
        elif num == 2:
            dropNullAndRow()
        elif num == 3:
            toTxt()
        elif num == 4:
            toExcel()
        elif num == 5:
            plot()
        elif num == 0:
            break
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()