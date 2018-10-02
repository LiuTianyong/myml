import pandas as pd

dataset = pd.read_csv('food_info.csv')
# print(type(dataset))
# #前三行
# print(dataset.head(3))
# #后四行
# print(dataset.tail(4))
# #所有列名
# print(dataset.columns)
# #数据规模
# print(dataset.shape)
# #拿到第i行值
# print(dataset.loc[0])
# print(dataset.loc[0:3])
#以列取
#print(dataset['NDB_No'])

#查找以g为单位的列
# col_names = dataset.columns.tolist()
# gram_columns = []
#
# for c in col_names:
#     if c.endswith("(g)"):
#         gram_columns.append(c)
# print(dataset[gram_columns])

dataset.sort_values('Water_(g)',inplace=True)
print(dataset)
