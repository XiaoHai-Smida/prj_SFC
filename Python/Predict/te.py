import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # 回归任务（若分类则用RandomForestClassifier）
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 假设数据格式：X为特征矩阵（n_samples×n_features），y为目标变量
# 示例数据生成（替换为实际数据）
X = np.random.rand(100, 30)  # 100个样本，30个特征
y = np.random.rand(100)  # 连续型目标变量

# 数据标准化（按需启用）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 参数按需调整

# 留一法交叉验证
loo = LeaveOneOut()
mse_scores = []
r2_scores = []
y_true = []
y_pred = []

for train_idx, test_idx in loo.split(X_scaled):
    # 分割训练集与测试集
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测
    pred = rf.predict(X_test)

    # 保存结果
    y_true.append(y_test)
    y_pred.append(pred)
    mse_scores.append(mean_squared_error(y_test, pred))
    r2_scores.append(r2_score(y_test, pred))

# 计算总体指标
mean_mse = np.mean(mse_scores)
mean_r2 = np.mean(r2_scores)

# 输出结果
print(f"LOOCV Results ({len(y_true)} samples):")
print(f"Mean MSE: {mean_mse:.4f}")
print(f"Mean R²: {mean_r2:.4f}")

# 查看每个样本的预测与真实值对比
results_df = pd.DataFrame({
    'True': y_true,
    'Predicted': y_pred
})
print("\nSample-level predictions:")
print(results_df.head(10))
