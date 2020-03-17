# -*- coding: utf-8 -*-

import os
import pandas, matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False 


month_dict = {'January': '01', 'February': '02', 'March': '03', 'April': '04', 
         'May': '05', 'June': '06', 'July': '07', 'August': '08',
         'September': '09', 'October': '10', 'November': '11', 'December': '12'}
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']


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


def newDf():
    df = pandas.read_csv(inputFile('.csv'))
    data = {'DATA': [], 'SOI': []}
    for j in range(len(df)):
        item = df.iloc[j]
        for i in month:
            data['DATA'].append('{}-{}-01'.format(int(item['Year']), month_dict[i]))
            data['SOI'].append(item[i])
            
    new_df = pandas.DataFrame(data)
    new_df = new_df.dropna()  # 删除空值
    print('#数据处理完毕，保存txt文件...')
    out_path = outputFile('.txt')
    new_df.to_csv(out_path, index=0)
    print('=>保存完毕（{}）'.format(out_path))


def statistics():
    df = pandas.read_csv(inputFile('.txt'))
    print('#####################################')
    print('SOI max: {}'.format(df['SOI'].max()))
    print('SOI min: {}'.format(df['SOI'].min()))
    print('SOI mean: {}'.format(df['SOI'].mean()))
    print('#####################################')
    

def disperse():
    df = pandas.read_csv(inputFile('.txt'))
    mi = df['SOI'].min()
    ma = df['SOI'].max()
    dis = pandas.cut(df['SOI'], [mi, 0, ma], labels=['NinoRelate', 'LaNinoRelate'])  # 离散化处理
    df['Label'] = dis  # 增加新列
    print('#离散化处理完毕，保存csv文件...')
    out_path = outputFile('.csv')
    df.to_csv(out_path, index=0)
    print('=>保存完毕（{}）'.format(out_path))
    data = [dis.value_counts()['NinoRelate'], dis.value_counts()['LaNinoRelate']]
    plt.figure(figsize=(6,6))
    plt.pie(data, labels=['NinoRelate', 'LaNinoRelate'], autopct='%1.1f%%', explode=[0.01, 0.01])
    plt.title('NinoRelate, LaNinoRelate离散饼状图')
    print('#饼状图绘制完毕，保存png文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=300)
    # plt.show()
    print('=>保存完毕（{}）'.format(out_path))
    
    
def soiPlot():
    df = pandas.read_csv(inputFile('.csv'))
    plt.figure(figsize=(10, 8))
    plt.plot(df['SOI'], color='blue', label='SOI')
    plt.legend()
    years = range(1866, 2018, 10)
    plt.xticks(range(0, len(df), 120), years)
    plt.xlabel('年份')
    plt.ylabel('SOI')
    plt.title('1866-2018 年南方涛动指数 SOI 折线图')
    print('#SOI折线图绘制完毕，保存png文件...')
    out_path = outputFile('.png')
    plt.savefig(out_path, dpi=400)
    # plt.show()
    print('=>保存完毕（{}）'.format(out_path))
    
    
def main():
    while True:
        print('\n**********************************************')
        print('请输入需要执行的操作的编号')
        print('1.读取csv文件，抽取DATA和SOI并保存到txt')
        print('2.读取txt文件统计SOI的min, max, mean')
        print('3.读取txt文件统计离散值及其饼状图并保存到csv文件')
        print('4.读取csv文件并绘制SOI的折线图')
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
            soiPlot()
        elif num == 0:
            break
        else:
            print('*ERROR:编号输入不正确')


if __name__ == '__main__':
    main()





















