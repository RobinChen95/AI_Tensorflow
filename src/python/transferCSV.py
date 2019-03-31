# coding=utf-8

import csv

import numpy as np

csv_file = csv.reader(open('small.csv', 'r'))
rate_list = []
movie_seq = 0

# 后900个数据不要了
for rate in csv_file:
    if rate[1] == "101":
        break
    if rate[1] != str(movie_seq):
        rate_list.append([rate[2]])
        movie_seq = int(rate[1])
    else:
        rate_list[movie_seq - 1].append(rate[2])

# 找到最长的行的长度，对长度不足的行进行0填充，形成矩阵
length = 0
for list_item in rate_list:
    if len(list_item) > length:
        length = len(list_item)
for list_item in rate_list:
    if len(list_item) < length:
        while (len(list_item) != length):
            list_item.append(0)
# 存储一个int类型的rate_list,保存第1步结果矩阵
int_rate_list = []
for i in range(len(rate_list)):
    int_rate_list.append([])
    for j in range(len(rate_list[i])):
        int_rate_list[i].append(int(rate_list[i][j]))


"""
row = 1
col = 1
# 将矩阵转换为老师规定的形式
handled_list = []
for list_item in rate_list:
    for num in list_item:
        handled_list.append([str(row), str(col), num])
        col += 1
    row += 1
    col = 1

# 传入标识符'w'或者'wb'表示写文本文件或写二进制文件
f = open("out.csv", "wb")
for list_item in handled_list:
    for chars in list_item:
        f.write(str(chars))
        f.write(" ")
    f.write("\n")
"""


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


# 余弦相似度矩阵,保存第二步结果矩阵
cos_sim_matrix = []
for i in range(len(rate_list)):
    cos_sim_matrix.append([])
    for j in range(len(rate_list)):
        cos_sim_matrix[i].append(cos_sim(int_rate_list[i], int_rate_list[j]))

# 设置numpy全部输出
np.set_printoptions(threshold=1e10)
# 第三步的结果矩阵
result = np.matmul(cos_sim_matrix, int_rate_list)

f = open("3.csv", "wb")
for list_item in result:
    for chars in list_item:
        f.write(str(chars))
        f.write(" ")
    f.write("\n")

for i in range(len(int_rate_list)):
    for j in range(len(int_rate_list[i])):
        if int_rate_list[i][j] > 0:
            result[i][j] = 0


# 传入标识符'w'或者'wb'表示写文本文件或写二进制文件
f = open("temp.csv", "wb")
for list_item in result:
    for chars in list_item:
        f.write(str(chars))
        f.write(" ")
    f.write("\n")
