import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pytorchtools import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import os
# 1. 数据加载
class CustomDataset(Dataset):
    def __init__(self, data):
        # 读取数据
        self.features = torch.tensor(data[:, :-1], dtype=torch.float32)  # 特征
        self.labels = torch.tensor(data[:, -1], dtype=torch.long)  # 标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 2. 定义模型
class ThreeClassNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 3. CUDA设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 5. 初始化模型、损失函数和优化器
input_size = 12  # 特征的维度
hidden_size = 64  # 隐藏层节点数，可以调整
output_size = 3  # 三分类问题



# 4. 加载数据
file_path = './DATA/p1g1.txt'  # 替换为你的txt文件路径
data = np.loadtxt(file_path, delimiter='\t')
num_samples = data.shape[0]
results_save = []
train_loss_list = []
train_acc_list = []
val_label_list = []
val_pred_list = []



for sample in range(num_samples):
    val_cv = data[sample]
    train_cv = np.delete(data, sample, axis=0)

    dataset = CustomDataset(train_cv)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # train model
    num_epochs = 300
    model = ThreeClassNN(input_size, hidden_size, output_size).to(device)  # 将模型转移到GPU
    criterion = nn.CrossEntropyLoss()  # 适用于分类任务的交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=30, verbose=True, delta=0.000,
                                   path='.\\checkpoint\\CV%s' % (sample+1)+ "_model.pth")

    train_loss_epoch_list = []
    train_acc_epoch_list = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        train_size = int(0.8 * len(dataset))  # 80% 作为训练集
        test_size = len(dataset) - train_size  # 剩下的作为测试集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for inputs, labels in train_loader:
            # 将数据转移到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        test_loss, test_samples = 0., 0.
        for inputs, labels in test_loader:
            model.eval()
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 0)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_samples += labels.size(0)
        # val_loss = 0
        # correct_val = 0
        # for inputs, labels in test_loader:
        #     model.eval()
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     with torch.no_grad():
        #         outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     _, predicted = torch.max(outputs, 1)
        #     val_loss += loss.item() * inputs.size(0)
        #     correct_val += (predicted == labels).sum().item()


        avg_loss = total_loss / total_samples
        avg_test_loss = test_loss / test_samples
        accuracy = correct_predictions / total_samples
        print(f'CV [{sample+1}/55], Epoch [ {epoch + 1}/{num_epochs}], train_loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, test_loss: {avg_test_loss:.4f}')
        train_loss_epoch_list.append(avg_loss)
        train_acc_epoch_list.append(accuracy)

        early_stopping(test_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
    train_loss_list.append(train_loss_epoch_list)
    train_acc_list.append(train_acc_epoch_list)

    model.eval()
    val_inputs = torch.tensor(val_cv[:-1], dtype=torch.float32)
    val_labels = torch.tensor(val_cv[-1], dtype=torch.long)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
    with torch.no_grad():
        outputs = model(val_inputs)
    # loss = criterion(outputs, val_labels)
    _, predicted = torch.max(outputs, 0)

    val_label_list.append(val_labels.item())
    val_pred_list.append(predicted.item())


# 计算精确率、召回率、F1分数（每一类的精确率、召回率、F1分数）
precision = precision_score(val_label_list, val_pred_list, average='macro')  # None 表示每个类别的精确率
recall = recall_score(val_label_list, val_pred_list, average='macro')
f1 = f1_score(val_label_list, val_pred_list, average='macro')
acc_score = accuracy_score(val_label_list, val_pred_list)

print(f"| Precision | Recall | F1 | ACC |")
print(f"{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{acc_score:.4f}\n")

# with open('results.txt', 'w') as f:
#     f.write("True_labels\tPredicted\n")
#     # 遍历所有保存的结果
#     for i in range(len(results_save)):
#         f.write(f"{results_save[i]}\n")  # 写入到文件


# def loss_plot(data):
#     # plot loss figure
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Train Loss', color='blue')
#     plt.plot(val_losses, label='Val Loss', color='orange')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(checkpoint_dir, "loss_plot.png"), dpi=300)
#     plt.show()
#
#     # 7. 保存模型
#     # torch.save(model.state_dict(), 'three_class_nn.pth')
#
# loss_plot(train_loss_list)

# 将矩阵存入txt文件
with open('./PLOTS/loss.txt', 'w') as f:
    for row in train_loss_list:
        row_data = '\t'.join(map(str, row))  # 将每行的数字转换为字符串，并用制表符（\t）分隔
        f.write(row_data + '\n')  # 写入每一行，行尾添加换行符

with open('./PLOTS/acc.txt', 'w') as f:
    for row in train_acc_list:
        row_data = '\t'.join(map(str, row))  # 将每行的数字转换为字符串，并用制表符（\t）分隔
        f.write(row_data + '\n')  # 写入每一行，行尾添加换行符
