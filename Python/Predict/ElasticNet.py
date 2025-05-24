import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# 1. 加载数据（格式与随机森林示例一致）
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None)
data = pd.read_csv('../../R/csv/PI_data.csv')
print(X.shape, data.shape)
# X = pd.concat([X, data.iloc[:,:7]],axis=1 )
y = data.iloc[:,7:10]

# 标准化特征（仅在训练集上拟合）
mean_X = X.mean(axis=0)
std_X = X.std(axis=0)
scaled_X = (X - mean_X) / std_X

mean_y = y.mean(axis=0)
std_y = y.std(axis=0)
scaled_y = (y - mean_y) / std_y

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size=0.2, random_state=42)





def model1():
    # 初始化弹性网络模型（增加最大迭代次数确保收敛）
    model = ElasticNet(max_iter=10000)

    # 设置参数网格
    param_grid = {
        'alpha': np.logspace(-4, 2, 10),  # 正则化强度：0.0001到100
        'l1_ratio': np.linspace(0.1, 0.9, 9)  # L1/L2比例：10%到90%
    }

    # 配置网格搜索（5折交叉验证，并行计算）
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("最佳参数组合：", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # 测试集评估
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差(MSE): {mse:.4f}")
    print(f"模型系数数量：{np.sum(best_model.coef_ != 0)}")  # 非零系数数量

    # 可选：分析交叉验证结果
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(cv_results[['param_alpha', 'param_l1_ratio', 'mean_test_score']].head())

def model2():
    model = ElasticNet(random_state=0)
    model.fit(X, y)
    print(model.coef_)
