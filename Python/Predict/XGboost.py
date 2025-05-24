# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 加载数据（格式与随机森林示例一致）
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None)
data = pd.read_csv('../../R/csv/PI_data.csv')
print(X.shape, data.shape)
# X = pd.concat([X, data.iloc[:,:7]],axis=1 )
y = data.iloc[:,7:10]

# 2. 数据预处理
X.fillna(X.mean(), inplace=True)  # 处理缺失值

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. 定义XGBoost模型和参数网格
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # 回归任务
    random_state=42,
    n_jobs=-1
)

# 超参数搜索空间（关键参数组合）
param_grid = {
    'n_estimators': [100, 200, 300],    # 树的数量
    'max_depth': [3, 5, 7],             # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'subsample': [0.8, 1.0],            # 样本子采样比例
    'colsample_bytree': [0.8, 1.0]      # 特征子采样比例
}

# 5. 网格搜索调优
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # 优化目标为MSE
    cv=5,                              # 5折交叉验证
    verbose=2,                         # 输出调优过程
    n_jobs=-1                          # 并行计算
)
grid_search.fit(X_train, y_train)

# 6. 输出最优参数
print("最优超参数组合:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# 7. 评估测试集性能
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"测试集均方误差 (MSE): {mse:.2f}")
print(f"测试集R²分数: {r2:.2f}")

# 8. 特征重要性分析
importance = best_xgb.get_booster().get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x, reverse=True)

print("\n特征重要性排名 (Top 10):")
for feat, score in sorted_importance[:10]:
    print(f"特征 {feat}: {score:.1f}")

# 9. 保存模型
# joblib.dump(best_xgb, 'optimized_xgboost.pkl')

# 10. 可选：SHAP值解释（需安装shap库）
# import shap
# explainer = shap.TreeExplainer(best_xgb)
# shap_values = explainer.shap_values(X_train)
# shap.summary_plot(shap_values, X_train, plot_type="bar")
