import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from pytorchtools import EarlyStopping
import os
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# data loading
X = pd.read_csv('../../R/csv/sfc_sem6.csv', header=None) #sfc
data = pd.read_csv('../../R/csv/PI_data.csv')
print(X.shape, data.shape)

## use specific regions
regions = [x-1 for x in [11,44,45,42,82,28]]
# ### regions = [x-1 for x in [11,44,45]]
X = X.iloc[:,regions]

## add envs
X = pd.concat([X, data.iloc[:,:7]],axis=1)

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
batch_size = 16
num_outputs = 3
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
    # loss_value = 1 - cov / (std_pred * std_true + 1e-8)
    return 1 - cov / (std_pred * std_true + 1e-8)

def R2Loss(y_pred, y_true):
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return torch.mean(1 - r2)


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
            val_loss = mse_ratio * criterion(y_test, y_pred) + (1-mse_ratio) * pearson_loss(y_test, y_pred)

        train_loss = np.mean(train_losses)

        # early_stopping(val_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练

        print('| CV{:3d} | epoch{:3d} | time: {:5.2f}s | train_loss {:5.2f} | val_loss {:5.2f} |'.format(
            cv, epoch, (time.time() - epoch_start_time), train_loss, val_loss))

    # output cv results
    model.eval()
    with torch.no_grad():
        tensor_cvX = torch.FloatTensor(cv_X).unsqueeze(0).to(device)
        tensor_cvy = torch.FloatTensor(cv_y).unsqueeze(0).to(device)
        y_pred = model(tensor_cvX)
        # mse = criterion(y_pred, tensor_cvy)
        predictions.append(y_pred.squeeze().cpu().numpy())

# output results
out_dir = os.path.abspath(os.path.join('./OUT/', 'results_%s' % time.strftime("%m-%d-%H_%M",time.localtime())))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

predictions = np.array(predictions)
final_predictions = predictions # * std_y + mean_y
r,p,r2, mse = np.zeros([3,1]), np.zeros([3,1]), np.zeros([3,1]), np.zeros([3,1])

r[0],p[0] = pearsonr(final_predictions[:,0], y[:,0])
r[1],p[1] = pearsonr(final_predictions[:,1], y[:,1])
r[2],p[2] = pearsonr(final_predictions[:,2], y[:,2])
print(f'pearson r:{r}\n')
r2[0] = r2_score(final_predictions[:,0], y[:,0])
mse[0] = mean_squared_error(final_predictions[:,0], y[:,0])
r2[1] = r2_score(final_predictions[:,1], y[:,1])
mse[1] = mean_squared_error(final_predictions[:,1], y[:,1])
r2[2] = r2_score(final_predictions[:,2], y[:,2])
mse[2] = mean_squared_error(final_predictions[:,2], y[:,2])

res = np.hstack((mse,r2,r,p))
out_file = os.path.join(out_dir, 'results.txt')
np.savetxt(out_file, res)
# mse r2 r p

plt.scatter(y[:,0], final_predictions[:, 0], c='blue')
plt.scatter(y[:,1], final_predictions[:, 1], c='red')
plt.scatter(y[:,2], final_predictions[:, 2], c='green')
plt.show()
