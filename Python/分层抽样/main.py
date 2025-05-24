import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# data loading
df = pd.read_csv('../../R/csv/PI_data.csv')
df2 = pd.DataFrame(None, columns=["a1","b1",'b2','b3','pma','sex'])
df2['a1'], df2['pma'], df2['sex'] = df['a1'], df['c1'], df['c2']
df2['b1'] = df[['b1', 'b4']].mean(axis=1)
df2['b2'] = df[['b2', 'b5']].mean(axis=1)
df2['b3'] = df[['b3', 'b6']].mean(axis=1)
df2['key'] = range(len(df2))
# 定义分层抽样函数
# def stratified_sampling(df, n_groups=3):
"""按性别B和年龄A分层，每个层内根据C中位数划分高低组"""
    # 将连续年龄分为n_groups个区间
# df = df2
# n_groups = 3
def func(df,n_groups,type):
    df['age_group'] = pd.qcut(df['pma'], q=n_groups, labels=[f'age_{i + 1}' for i in range(n_groups)])

    # 初始化分组容器
    high_list, low_list = [], []

    # 遍历每个性别和年龄层的组合
    for (gender, age_group), group in df.groupby(['sex', 'age_group']):
        if len(group) < 5:  # 忽略样本过少的层（可调整阈值）
            continue

        # 按C的中位数划分
        median_c = group[type].median()
        high = group[group[type] >= median_c]
        low = group[group[type] < median_c]

        # 平衡样本量：随机抽样使两组数量相等
        min_size = min(len(high), len(low))
        high_list.append(high.sample(min_size, random_state=4))
        low_list.append(low.sample(min_size, random_state=4))

    # 合并所有层的结果
    high_group = pd.concat(high_list)
    low_group = pd.concat(low_list)
    return high_group, low_group

env='a1'
high_group, low_group = func(df2,3, env)

plt.hist(high_group['pma'], bins=30, alpha=0.7, color='blue', edgecolor='black',density=True)
plt.hist(low_group['pma'], bins=30, alpha=0.7, color='red', edgecolor='black',density=True)
plt.show()

r = stats.ttest_ind(high_group[env], low_group[env])
print(high_group.mean(axis=0))
print(low_group.mean(axis=0))
print(r)

# save index
res = pd.DataFrame({
    'L': low_group['key'].values +1,
    'H': high_group['key'].values +1
                   })
res.to_csv(f"output_{env}.txt", index=False, header=True, encoding='utf-8', sep='\t')
