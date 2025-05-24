import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from pytorchtools import EarlyStopping
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# data loading
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None) #sfc
data = pd.read_csv('../../R/csv/PI_data.csv')
print(X.shape, data.shape)

# ## use specific regions
regions = [x-1 for x in [11,44,45,42,82,28]]
X = X.iloc[:,regions]
#
# ## add envs
X = pd.concat([X, data.iloc[:,:7]],axis=1)

y = data.iloc[:,9]
X = X.to_numpy()
y = y.to_numpy()

# 标准化特征（仅在训练集上拟合）
mean_X = X.mean(axis=0)
std_X = X.std(axis=0)
scaled_X = (X - mean_X) / std_X
# mean_y = y.mean(axis=0)
# std_y = y.std(axis=0)
# scaled_y = (y - mean_y) / std_y
scaled_y = y

# model parameters
num_samples, num_features = X.shape
batch_size = 32
num_outputs = 1
num_epoch = 500
mse_ratio = 1

class RegressionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32, bias=False)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


# Cross validation
def getbatchdata(features, labels, batch_size):
    num_samples = features.shape[0]
    indeces = np.arange(num_samples)
    np.random.shuffle(indeces)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indeces[start_idx:end_idx]
        yield features[batch_indices], labels[batch_indices]
def pearson_loss(y_pred, y_true):
    y_pred_centered = y_pred - torch.mean(y_pred)
    y_true_centered = y_true - torch.mean(y_true)
    cov = torch.mean(y_pred_centered * y_true_centered)
    std_pred = torch.std(y_pred)
    std_true = torch.std(y_true)
    return 1 - cov / (std_pred * std_true + 1e-8)

mse_losses = []
predictions = []
for cv in range(num_samples):
    cv_X = scaled_X[cv]
    cv_y = scaled_y[cv]
    dataset_X = np.delete(scaled_X, cv, axis=0)
    dataset_y = np.delete(scaled_y, cv, axis=0)

    tensor_X = torch.FloatTensor(dataset_X).to(device)
    tensor_y = torch.FloatTensor(dataset_y).unsqueeze(1).to(device)

    X_train, X_test, y_train, y_test = train_test_split(tensor_X, tensor_y, test_size=0.2, random_state=42)

    # model definition
    model = RegressionNet(input_size=num_features, output_size=num_outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    save_path = f"./checkpoints/EarlyStoppingCheckpoint_{cv:03d}.pth"
    early_stopping = EarlyStopping(patience=20, verbose=False, delta=0.000,
                                   path=save_path)


    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        # define batch data
        batch_data = getbatchdata(tensor_X, tensor_y, batch_size)

        train_losses = []

        model.train()
        for batch_idx, (X_batch, y_batch) in enumerate(batch_data):
            outputs = model(X_batch)
            loss = mse_ratio * criterion(outputs, y_batch) + (1-mse_ratio) * pearson_loss(outputs, y_batch)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            val_loss = criterion(y_pred, y_test)

        train_loss = np.mean(train_losses)

        early_stopping(val_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        print('| CV{:3d} | epoch{:3d} | time: {:5.2f}s | train_loss {:5.2f} | val_loss {:5.2f} |'.format(
            cv, epoch, (time.time() - epoch_start_time), train_loss, val_loss))

    # output cv results
    model.eval()
    with torch.no_grad():
        tensor_cvX = torch.FloatTensor(cv_X).unsqueeze(0).to(device)
        tensor_cvy = torch.FloatTensor([cv_y]).unsqueeze(0).to(device)
        y_pred = model(tensor_cvX)
        mse = criterion(y_pred, tensor_cvy)
        predictions.append(y_pred.squeeze().cpu().numpy())

# output results
predictions = np.array(predictions)
final_predictions = predictions # * std_y + mean_y

# def pearson_correlation(output, target):
#     vx = output - np.mean(output) + 1e-8
#     vy = target - np.mean(target) + 1e-8
#     cost = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
#     return cost

r,p = pearsonr(final_predictions, y)
# r[1] = pearson_correlation(final_predictions[:,1], y[:,1])
# r[2] = pearson_correlation(final_predictions[:,2], y[:,2])

print(f'pearson r:{r}\np:{p}\n')

plt.scatter(y, final_predictions, c='blue')
# plt.scatter(y[:,1], final_predictions[:, 1], c='red')
# plt.scatter(y[:,2], final_predictions[:, 2], c='green')
plt.show()
