# -*- coding: utf-8 -*-

import os
import pandas, matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# 设置matplotlib可以显示中文
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False 


month_dict = {'January': '01', 'February': '02', 'March': '03', 'April': '04', 
         'May': '05', 'June': '06', 'July': '07', 'August': '08',
         'September': '09', 'October': '10', 'November': '11', 'December': '12'}
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']


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


def newDf():
    # 任务一，抽取时间为单独列并保存DATA和NAO到新数据集，新数据集保存为txt文件
    df = pandas.read_csv(inputFile('.csv'))
    data = {'DATA': [], 'NAO': []}
    for j in range(len(df)):
        item = df.iloc[j]
        for i in month:
            data['DATA'].append('{}-{}-01'.format(int(item['Year']), month_dict[i]))
            data['NAO'].append(item[i])
            
    new_df = pandas.DataFrame(data)
    new_df = new_df.dropna()  # 删除空值
    new_df['NAO'][new_df['NAO'] == -99.99] = -1  # 处理异常值
    print('#数据处理完毕，保存txt文件...')
    out_path = outputFile('.txt')
    new_df.to_csv(out_path, index=0)
    print('=>保存完毕（{}）'.format(out_path))


def statistics():
    # 任务二，统计NAO的最大值，最小值和均值
    df = pandas.read_csv(inputFile('.txt'))
    print('###################################')
    print('NAO max: {}'.format(df['NAO'].max()))  # 最大值
    print('NAO min: {}'.format(df['NAO'].min()))  # 最小值
    print('NAO mean: {}'.format(df['NAO'].mean()))  # 均值
    print('###################################')
    

def disperse():
    # 任务三，离散化处理，并将离散值绘制为饼图
    df = pandas.read_csv(inputFile('.txt'))  # 读取文件
    mi = df['NAO'].min()  # NAO列的最小值
    ma = df['NAO'].max()  # 最大值
    dis = pandas.cut(df['NAO'], [mi, 0, ma], labels=['ColdRelate', 'WarmRelate'])  # 离散化
    df['Label'] = dis  # 赋值新列
    print('#离散化处理完毕，保存csv文件...')
    out_path = outputFile('.csv')  # 输入保存文件路径
    df.to_csv(out_path, index=0)  # 保存文件
    print('=>保存完毕（{}）'.format(out_path))
    data = [dis.value_counts()['ColdRelate'], dis.value_counts()['WarmRelate']]  # value_count:统计离散值有多少个
    plt.figure(figsize=(6,6))  # 画布尺寸
    plt.pie(data, labels=['ColdRelate', 'WarmRelate'], autopct='%1.1f%%', explode=[0.01, 0.01])  # 绘图
    plt.title('ColdRelate, WarmRelate离散饼状图')  # 图标题
    print('#饼状图绘制完毕，保存png文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=300)  # 保存图到本地
    # plt.show()
    print('=>保存完毕（{}）'.format(out_path))
    
    
def NAOPlot():
    # 任务四，绘制NAO的折线图
    df = pandas.read_csv(inputFile('.csv'))
    plt.figure(figsize=(10, 8))
    plt.plot(df['NAO'], color='blue', label='NAO')
    plt.legend()  # 显示图例
    years = range(1824, 2018, 10)
    plt.xticks(range(0, len(df), 120), years)  # x轴刻度
    plt.xlabel('年份')  # x轴标签
    plt.ylabel('NAO')   # y轴标签
    plt.title('1824-2018 年北极涛动指数 NAO 折线图')  # 图标题
    print('#NAO折线图绘制完毕，保存png文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=400)  # 保存
    # plt.show()
    print('=>保存完毕（{}）'.format(out_path))
    
    
def main():
    while True:
        print('\n**********************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件，抽取DATA和NAO并保存到txt')
        print('2.读取txt文件统计NAO的min, max, mean')
        print('3.读取txt文件统计离散值及其饼状图并保存到csv文件')
        print('4.读取csv文件并绘制NAO的折线图')
        print('0.退出')
        print('**********************************************')
        num = input('>>> ')
        try:
            num = int(num)
        except:
            print('*ERROR:请输入数字编号')
        if num == 1:
            newDf()
        elif num == 2:
            statistics()
        elif num == 3:
            disperse()
        elif num == 4:
            NAOPlot()
        elif num == 0:
            break
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()