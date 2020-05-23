import numpy as np
import itertools as it
import datetime

""""
1) 从 0-47 中抽取 k 个组合 sample_combin
2) 将 sample_combin 中的元素依次转换成三维 (3,4,4) 中的对应坐标 coord_3
3) 将 coord_3 再依次转换成 0/1 二值形式的 18 维向量 vector_18,并依次添加到列表 vector 做去冗余操作
4) 把 vector 映射到 1-2^18 对应数值 num,并依次添加到集合 num_set 筛选重复的数
5) 最后 num_set 的长度即为最终要求的结果
"""
# 数值转换成三维(3,4,4)
def turn_48_to_coord_3(num):
    for i in range(3):
        for j in range(4):
            for k in range(4):
                if i * 16 + j * 4 + k == num:
                    return [i + 1,j + 1,k + 1]

# 三维(3,4,4)转换成 18 维向量
def coord_3_to_18(coord_3):
    vector_18 = np.zeros([2,3,3])
    # 如果色泽为 *
    if coord_3[0] == 3:
        coord_3[0] = [1, 2]
    else:
        coord_3[0] = [coord_3[0]]
    # 如果根蒂为 *
    if coord_3[1] == 4:
        coord_3[1] = [1, 2, 3]
    else:
        coord_3[1] = [coord_3[1]]
    # 如果敲声为 *
    if coord_3[2] == 4:
        coord_3[2] = [1, 2, 3]
    else:
        coord_3[2] = [coord_3[2]]
    for x in coord_3[0]:
        for y in coord_3[1]:
            for z in coord_3[2]:
                # 映射到 18 维向量的值为 1 表示相应特征
                vector_18[x-1][y-1][z-1] = 1
    return vector_18


# 获得 0-48 数值转换成 18 维向量的结果
def get_48_to_18(num):
    coord_3 = turn_48_to_coord_3(num)
    vector_18 = coord_3_to_18(coord_3)
    return vector_18
def main(k):
    num_set = []
    # 从 0-47 中抽取 k 个组合
    for sample_combin in it.combinations(range(48),k):
        vector = []
        for i in range(k):
            vector_18 = get_48_to_18(sample_combin[i])
            vector.append(vector_18)
        vector = np.array(vector)
        vector = vector.any(axis=0) # 去冗余操作:按第一个轴方向取或
        vector = np.reshape(vector,[18])
        vector = vector.tolist()
        num = 0
        for i in range(18):
            num += 2 ** i * vector[i] # 0/1 二值 18 维映射成 1-2^18 十进制
        num_set.append(num)
        if len(num_set) > 5000000:
            num_set = list(set(num_set)) # 长度大于 500W 时取一次集合,防止数组太长导致程序崩溃
    # 最后取一次集合
    num_set = list(set(num_set))
    end_time1 = datetime.datetime.now()
    print('k=%d时： %d examples' %(k, len(num_set)))
    print('   用时:', end_time1 - start_time1)

start_time0 = datetime.datetime.now()
for k in range(1,18):
    start_time1 = datetime.datetime.now()
    main(k)
end_time0 = datetime.datetime.now()
print('一共用时',end_time0 - start_time0)