# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 加载数据（假设数据为CSV格式，最后一列为测验分数）
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None)
data = pd.read_csv('../../R/csv/PI_data.csv')

# ## use specific regions
regions = [x-1 for x in [11,44,45,42,82,28]]
# # ### regions = [x-1 for x in [11,44,45]]
X = X.iloc[:,regions]

## add envs
X = pd.concat([X, data.iloc[:,:7]],axis=1)

y = data.iloc[:,7:10]

X = X.to_numpy()
y = y.to_numpy()
print(X.shape, data.shape)


num_samples, num_features = X.shape
predictions, importance = [], []
for cv in tqdm(range(num_samples)):
    cv_X = X[cv]
    cv_y = y[cv]
    dataset_X = np.delete(X, cv, axis=0)
    dataset_y = np.delete(y, cv, axis=0)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     dataset_X, dataset_y,
    #     test_size=0.2,
    #     random_state=42
    # )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(dataset_X, dataset_y)

    cv_X = np.reshape(cv_X, [1, -1])
    y_pred = rf.predict(cv_X)
    predictions.append(y_pred[0])

    fold_importance = rf.feature_importances_
    importance.append(fold_importance)


out_dir = os.path.abspath(os.path.join('./OUT/', 'results_%s' % time.strftime("%m-%d-%H_%M",time.localtime())))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
importance = np.array(importance)
predictions = np.array(predictions)

r,p,r2, mse = np.zeros([3,1]), np.zeros([3,1]), np.zeros([3,1]), np.zeros([3,1])

r[0],p[0] = pearsonr(predictions[:,0], y[:,0])
r[1],p[1] = pearsonr(predictions[:,1], y[:,1])
r[2],p[2] = pearsonr(predictions[:,2], y[:,2])

mse[0] = mean_squared_error(predictions[:,0], y[:,0])
mse[1] = mean_squared_error(predictions[:,1], y[:,1])
mse[2] = mean_squared_error(predictions[:,2], y[:,2])


res = np.hstack((mse,r))
rf_results = os.path.join(out_dir, 'rf_results.txt')
np.savetxt(rf_results, res)
rf_importance = os.path.join(out_dir, 'rf_importance.txt')
np.savetxt(rf_importance, importance)
# mse r2 r p

plt.scatter(y[:,0], predictions[:, 0], c='blue')
plt.scatter(y[:,1], predictions[:, 1], c='red')
plt.scatter(y[:,2], predictions[:, 2], c='green')
plt.show()

# # 6. 输出最优参数
# print("最优超参数组合:", grid_search.best_params_)
# best_rf = grid_search.best_estimator_
#
# # 7. 使用最优模型预测
# y_pred = best_rf.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"测试集均方误差 (MSE): {mse:.2f}")
# print(f"测试集R²分数: {r2:.2f}")
#
# # 8. 特征重要性分析
# importances = best_rf.feature_importances_
# feature_names = X.columns
# sorted_idx = np.argsort(importances)[::-1]  # 按重要性降序排列
#
# print("\n特征重要性排名 (Top 10):")
# for i in sorted_idx[:10]:
#     print(f"{feature_names[i]}: {importances[i]:.3f}")

# 9. 保存最优模型
# joblib.dump(best_rf, 'optimized_random_forest.pkl')


