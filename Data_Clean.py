import pandas as pd
import numpy as np

# 读取数据，使用空格作为分隔符
df = pd.read_csv('all-mias/info/info.csv', sep='\\s+', header=None)

# 替换空缺的值为NaN
df.replace('', np.nan, inplace=True)

# 保存为新的CSV文件，用逗号分隔
df.to_csv('all-mias/info/info_clean.csv', index=False, header=False)
