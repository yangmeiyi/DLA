import pandas as pd

# 创建示例数据
data1_path1 = "/home/dkd/Data_4TDISK/Luoyi/code_ly/CCRCC/resnet50_basline_ccrcc_error.csv"
data1_path2 = "/home/dkd/Data_4TDISK/Luoyi/code_ly/CCRCC/ccrcc_error.csv"
data1_path3 = "/home/dkd/Data_4TDISK/CCRCC/ccrcc_test.csv"


df1 = pd.read_csv(data1_path1)
df2 = pd.read_csv(data1_path2)
df3 = pd.read_csv(data1_path3)

# 通过 'ID' 列对 df1 和 df2 进行分组，并找到交集
intersection_ids = set(df1['patient']).intersection(df2['patient'])
print(len(intersection_ids))
# 在 df3 中删除包含在交集中的 'ID' 行
df3 = df3[~df3['ID'].isin(intersection_ids)]
# print(df3)
# exit()

df3.to_csv("/home/dkd/Data_4TDISK/CCRCC/ccrcc_test_1.csv", index=False, sep=",")
