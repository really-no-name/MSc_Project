import pandas as pd
import numpy as np

# 读取数据，使用空格作为分隔符
df = pd.read_csv('all-mias/info/info.csv', sep='\\s+', header=None)

# 在第一列的每个值前添加指定的字符串
prefix = ".pgm"
df[0] = df[0].astype(str) + prefix

# 删除第一列中重复值所在的行，保留第一次出现的行
df.drop_duplicates(subset=[0], keep='first', inplace=True)

# 替换空缺的值为NaN
df.replace('', np.nan, inplace=True)

# 保存为新的CSV文件，用逗号分隔
df.to_csv('all-mias/info/info_clean.csv', index=False, header=False)
