import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from pytorchtools import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# data loading
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None) #sfc

# rearrange nodes with optimized path
optimized_path = np.loadtxt('optimized_path.csv', delimiter=',')
X = X.iloc[:, optimized_path]

data = pd.read_csv('../../R/csv/PI_data.csv')
print(X.shape, data.shape)

# ## add envs
# X = pd.concat([X, data.iloc[:,:7]],axis=1)

y = data.iloc[:,7:10]
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
num_outputs = 3
num_epoch = 500
mse_ratio = 1
learning_rate = 5e-3

class RegressionCNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride):
        super(RegressionCNN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        def conv_output_size(L_in, kernel_size, stride, pool=2):
            L_out = (L_in - kernel_size) // stride + 1
            L_out = L_out // pool  # MaxPool
            L_out = (L_out - kernel_size) // stride + 1
            L_out = L_out // pool  # 第二次MaxPool
            return L_out

        self.feature_size =conv_output_size(
            input_size, kernel_size=self.kernel_size, stride=self.stride) *32

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.kernel_size, stride=self.stride, bias=False),
            nn.BatchNorm1d(16), nn.LeakyReLU(negative_slope=0.1), nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=self.stride, bias=False),
            nn.BatchNorm1d(32), nn.LeakyReLU(negative_slope=0.1), nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=32, out_features=output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.dense(x)
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
    tensor_y = torch.FloatTensor(dataset_y).to(device)

    X_train, X_test, y_train, y_test = train_test_split(tensor_X, tensor_y, test_size=0.2, random_state=42)

    # model definition
    model = RegressionCNN(input_size=num_features, output_size=num_outputs, kernel_size=5, stride=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        tensor_cvy = torch.FloatTensor(cv_y).unsqueeze(0).to(device)
        y_pred = model(tensor_cvX)
        mse = criterion(y_pred, tensor_cvy)
        predictions.append(y_pred.squeeze().cpu().numpy())

# output results
predictions = np.array(predictions)
final_predictions = predictions # * std_y + mean_y
r = np.zeros(3)

def pearson_correlation(output, target):
    vx = output - np.mean(output) + 1e-8
    vy = target - np.mean(target) + 1e-8
    cost = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return cost

r[0] = pearson_correlation(final_predictions[:,0], y[:,0])
r[1] = pearson_correlation(final_predictions[:,1], y[:,1])
r[2] = pearson_correlation(final_predictions[:,2], y[:,2])

print(f'pearson r:{r}\n')

plt.scatter(y[:,0], final_predictions[:, 0], c='blue')
plt.scatter(y[:,1], final_predictions[:, 1], c='red')
plt.scatter(y[:,2], final_predictions[:, 2], c='green')
plt.show()
