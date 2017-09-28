# -*-coding:utf-8-*-

import pandas as pd
import os


path = './data/'


def get_file_name_list(path):
    file_name = []
    for file in os.listdir(path):
        if file.split('.')[-1] == 'csv':
            file_name.append(file)

    return file_name


def read_file(file_name):
    return pd.read_csv(path+file_name, header=0).iloc[:, [0, 4, 6]]


def main():
    global path
    file_name_list = get_file_name_list(path)
    df1 = pd.read_csv(path+file_name_list[0], header=0)
    df2 = pd.read_csv(path+file_name_list[1], header=0)
    df = pd.merge(df1.iloc[:, [0, 4, 6]], df2.iloc[:, [0, 4, 6]], on=['Date'], how='outer')

    # get columns' name
    old_headers = map(lambda x: x.split('.')[0], file_name_list)
    new_headers = []
    for i in old_headers:
        new_headers.append(i+'_prices')
        new_headers.append(i+'_volume')
    new_headers.insert(0, 'Date')

    # data merge
    for file_name in file_name_list[2:]:
        dff = read_file(file_name)
        df=pd.merge(df, dff, on=['Date'], how='left')
    # print df.shape
    df.to_csv(path+'result.csv', index=None, header=new_headers)


def test():
    file_name = get_file_name_list(path)
    file_num = len(file_name)
    print file_num
    for file in file_name:
        with open(path+file) as f:
            ll = len(f.readlines())
            if ll != 5286:
                print file


if __name__ == '__main__':
    # get_file_name_list(path)
    main()
    # test()